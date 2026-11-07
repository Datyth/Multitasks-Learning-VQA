#!/usr/bin/env python
"""Build reports/ from saved experiment outputs, or estimate whether the budget fits.

  python src/report.py                                  # works with any subset of experiments
  python src/report.py --estimate --hours-available 20  # uses timings from the smoke runs
  python src/report.py --estimate --hours-available 20 --sec-per-step 12 --sec-per-example 1.5
  python src/report.py --config config.v2_1000train_500test.yaml   # multi-seed report (config has train_seeds)

Only measured values are reported. Experiments that were not run appear as "not run".
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXPERIMENTS, QUERY_TASKS, TASKS, VERIFY_TASKS, compute_metrics, load_config, read_jsonl, resolve,
                    resolve_path, run_dir, write_json)

BOOTSTRAP_SAMPLES = 2000
INVALID_RATE_LIMIT = 0.10  # above this, a decision would mostly reflect generation failures

# Which experiment supplies which target task in the main results table.
ROWS = {
    "E0 pretrained": {"e0": list(TASKS)},
    "E1 single-task": {f"e1_{t.lower()}": [t] for t in TASKS},
    "E2 joint": {"e2_joint": list(TASKS)},
    "E3 question-type": {"e3_query": ["T1", "T2"], "e3_verify": ["T3", "T4"]},
    "E4 operation grouping": {"e4_no_relate": ["T1", "T3"], "e4_relate": ["T2", "T4"]},
}
# Off-diagonal source->target pairs by what the two tasks share.
RELATIONS = {
    "same question type (T1-T2, T3-T4)": [("T1", "T2"), ("T2", "T1"), ("T3", "T4"), ("T4", "T3")],
    "same operation group (T1-T3, T2-T4)": [("T1", "T3"), ("T3", "T1"), ("T2", "T4"), ("T4", "T2")],
    "neither (T1-T4, T2-T3)": [("T1", "T4"), ("T4", "T1"), ("T2", "T3"), ("T3", "T2")],
}


# --- loading --------------------------------------------------------------------------

def load_evaluation(directory: Path, experiment: str, expected: int, warnings: list):
    """One complete evaluation {metrics, by_task: task -> {question_id: record}}, or None."""
    metrics_path, predictions_path = directory / "metrics.json", directory / "predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        return None
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    by_task = {t: {} for t in EXPERIMENTS[experiment]["eval_tasks"]}
    for record in read_jsonl(predictions_path):
        if record["task"] in by_task:
            by_task[record["task"]][record["question_id"]] = record
    sizes = {t: len(rows) for t, rows in by_task.items()}
    if not metrics.get("complete") or any(n != expected for n in sizes.values()):
        warnings.append(f"{directory}: ignored as incomplete (samples per task {sizes}, expected {expected}).")
        return None
    return {"metrics": metrics, "by_task": by_task}


def load_runs(cfg: dict):
    """Load complete evaluations only. Smoke output lives elsewhere and is never read here."""
    runs, warnings = {}, []
    runs_dir = resolve(cfg, "runs_dir")
    expected = cfg["evaluation"]["samples_per_task"]
    for experiment in EXPERIMENTS:
        loaded = load_evaluation(runs_dir / experiment, experiment, expected, warnings)
        if loaded:
            runs[experiment] = loaded
    hashes = {r["metrics"].get("metadata_sha256") for r in runs.values()}
    if len(hashes) > 1:
        warnings.append("Experiments used different dataset metadata (hash mismatch); comparisons are not valid.")
    models = {r["metrics"].get("model") for r in runs.values()}
    if len(models) > 1:
        warnings.append(f"Experiments used different models: {sorted(map(str, models))}.")
    trains = {json.dumps(r["metrics"].get("config", {}).get("train"), sort_keys=True) for r in runs.values()}
    if len(trains) > 1:
        warnings.append("Experiments used different training budgets or settings; see each metrics.json.")
    return runs, warnings


def row_tasks(row: str, runs: dict) -> dict:
    """task -> {question_id: record} for a results row; absent experiments leave tasks absent."""
    tasks = {}
    for experiment, targets in ROWS[row].items():
        if experiment in runs:
            for task in targets:
                tasks[task] = runs[experiment]["by_task"][task]
    return tasks


def row_metrics(row: str, runs: dict) -> dict:
    return compute_metrics([r for rows in row_tasks(row, runs).values() for r in rows.values()])


# --- analysis -------------------------------------------------------------------------

def directed_transfer(baseline: dict, single: dict) -> dict:
    """T[i][j] = Acc_j(M_i) - Acc_j(M_0); None where either value is unavailable."""
    matrix = {}
    for source in TASKS:
        matrix[source] = {}
        for target in TASKS:
            value, base = single.get(source, {}).get(target), baseline.get(target)
            matrix[source][target] = None if value is None or base is None else value - base
    return matrix


def joint_interference(joint: dict, single: dict) -> dict:
    """I_i = Acc_i(M_joint) - Acc_i(M_i); negative means joint training hurt task i."""
    result = {}
    for task in TASKS:
        value, base = joint.get(task), single.get(task, {}).get(task)
        result[task] = None if value is None or base is None else value - base
    return result


def transfer_by_relation(matrix: dict) -> dict:
    """Mean off-diagonal transfer per task relationship. Descriptive only (four pairs each)."""
    result = {}
    for name, pairs in RELATIONS.items():
        values = [matrix[s][t] for s, t in pairs if matrix[s][t] is not None]
        result[name] = {"mean": sum(values) / len(values) if values else None, "pairs": len(values)}
    return result


def _paired_differences(a: dict, b: dict):
    if set(a) != set(b) or not a:
        return None
    ids = sorted(a)
    return np.array([float(a[i]["correct"]) - float(b[i]["correct"]) for i in ids])


def paired_ci(a: dict, b: dict, seed: int):
    """Difference a-b with a paired bootstrap 95% CI over test questions; None if ids differ."""
    diffs = _paired_differences(a, b)
    if diffs is None:
        return None
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (BOOTSTRAP_SAMPLES, len(diffs)))].mean(axis=1)
    return {"diff": float(diffs.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def macro_paired_ci(a: dict, b: dict, tasks, seed: int):
    """Equal-task-weight difference, bootstrapped within each task."""
    if not all(t in a and t in b for t in tasks):
        return None
    rng = np.random.default_rng(seed)
    point, totals = [], np.zeros(BOOTSTRAP_SAMPLES)
    for task in tasks:
        diffs = _paired_differences(a[task], b[task])
        if diffs is None:
            return None
        point.append(diffs.mean())
        totals += diffs[rng.integers(0, len(diffs), (BOOTSTRAP_SAMPLES, len(diffs)))].mean(axis=1)
    totals /= len(tasks)
    return {"diff": float(np.mean(point)), "ci95": [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))]}


def cluster_paired_ci(a: dict, b: dict, images: dict, tasks, seed: int):
    """Equal-task-weight mean of a-b with a 95% CI from a paired bootstrap over images.

    a, b: task -> {question_id: correctness in [0, 1]} (a seed average is allowed); images: question_id ->
    image_id. Questions on the same image are resampled together, across tasks too, so their correlation
    widens the interval. With one question per image this is the ordinary question-level paired bootstrap.
    """
    if not all(t in a and t in b and a[t] and set(a[t]) == set(b[t]) for t in tasks):
        return None
    ids = {t: sorted(a[t]) for t in tasks}
    clusters = sorted({images[q] for t in tasks for q in ids[t]})
    index = {c: i for i, c in enumerate(clusters)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), (BOOTSTRAP_SAMPLES, len(clusters)))
    weights = np.zeros((BOOTSTRAP_SAMPLES, len(clusters)))
    np.add.at(weights, (np.arange(BOOTSTRAP_SAMPLES)[:, None], draws), 1.0)
    point, boot = [], []
    for task in tasks:
        diffs = np.array([a[task][q] - b[task][q] for q in ids[task]], dtype=float)
        w = weights[:, [index[images[q]] for q in ids[task]]]
        total = w.sum(axis=1)
        boot.append(np.where(total > 0, (w @ diffs) / np.maximum(total, 1), np.nan))
        point.append(diffs.mean())
    means = np.mean(boot, axis=0)
    means = means[np.isfinite(means)]
    return {"diff": float(np.mean(point)), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "clusters": len(clusters)}


def research_decision(e4_vs_e3, invalid_rate) -> tuple[str, str]:
    """Pre-specified rule, applied mechanically; a researcher must still review it."""
    if e4_vs_e3 is None:
        return "Pending", "E3 and E4 are not both complete, so no decision is made."
    low, high = e4_vs_e3["ci95"]
    interval = f"E4 minus E3 macro accuracy is {e4_vs_e3['diff']:+.3f} (95% CI {low:+.3f} to {high:+.3f})"
    if invalid_rate is not None and invalid_rate > INVALID_RATE_LIMIT:
        return "Inconclusive", f"Invalid generation rate {invalid_rate:.1%} exceeds {INVALID_RATE_LIMIT:.0%}. {interval}."
    if low > 0:
        return "Promising", f"{interval}; the interval excludes zero. Continue with larger experiments and more seeds."
    if high < 0:
        return "Not supported", f"{interval}; the interval is entirely negative. Reconsider this grouping direction."
    return "Inconclusive", f"{interval}; the interval includes zero. Improve task design or repeat with more seeds."


# --- outputs --------------------------------------------------------------------------

def _fmt(value, signed=False, missing="not run") -> str:
    if value is None:
        return missing
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _fmt_ci(entry, missing="not run") -> str:
    if entry is None:
        return missing
    return f"{entry['diff']:+.3f} [{entry['ci95'][0]:+.3f}, {entry['ci95'][1]:+.3f}]"


def _table(header: list, rows: list) -> list[str]:
    lines = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows]


def plot_transfer_heatmap(matrix: dict, path: Path) -> None:
    """Diverging blue/red around a neutral midpoint at zero, every cell labeled, missing cells marked."""
    surface, ink, muted = "#fcfcfb", "#1a1a19", "#6b6b66"
    values = np.array([[np.nan if matrix[s][t] is None else matrix[s][t] for t in TASKS] for s in TASKS], dtype=float)
    limit = max(0.05, float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else 0.05
    cmap = LinearSegmentedColormap.from_list("transfer", ["#e34948", "#f0efec", "#2a78d6"])
    cmap.set_bad("#e6e5e1")

    fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor=surface)
    ax.set_facecolor(surface)
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit,
                         edgecolors=surface, linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(4) + 0.5, TASKS, color=ink)
    ax.set_yticks(np.arange(4) + 0.5, TASKS, color=ink)
    ax.set_xlabel("Target task j", color=muted, labelpad=8)
    ax.set_ylabel("Source task i (adapter trained on)", color=muted)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row, source in enumerate(TASKS):
        for col, target in enumerate(TASKS):
            value = matrix[source][target]
            ax.text(col + 0.5, row + 0.5, "n/a" if value is None else f"{value:+.3f}", ha="center", va="center",
                    color=muted if value is None else ink, fontsize=11, fontweight="bold" if row == col else "normal")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Accuracy change vs pretrained model", color=muted)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=muted)
    fig.suptitle("Directed transfer  Acc_j(M_i) - Acc_j(M_0)", color=ink, fontsize=12, y=0.98)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def write_results_csv(path: Path, table: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", *TASKS, "macro_avg", "query_macro", "verify_macro", "pooled", "invalid_rate", "status"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for row, m in table.items():
            per_task = m["per_task"]
            status = "complete" if len(per_task) == 4 else ("partial" if per_task else "not run")
            cells = [per_task[t]["accuracy"] if t in per_task else "" for t in TASKS]
            extra = [m["macro_accuracy"], m["query_macro_accuracy"], m["verify_macro_accuracy"], m["pooled_accuracy"],
                     m["invalid_rate"]]
            writer.writerow([row, *cells, *["" if v is None else v for v in extra], status])


def write_transfer_csv(path: Path, matrix: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source\\target", *TASKS])
        for source in TASKS:
            writer.writerow([source, *["" if matrix[source][t] is None else matrix[source][t] for t in TASKS]])


def _training_row(experiment: str, directory: Path) -> list:
    path, status, seconds, gpu = directory / "run_config.json", "not run", None, "n/a"
    if path.exists():
        with open(path) as fh:
            meta = json.load(fh)
        status, seconds = meta.get("status", "unknown"), meta.get("wall_seconds")
        gpu = meta.get("swift_env", {}).get("CUDA_VISIBLE_DEVICES", "n/a")
    log = _read_training_log(directory)
    return [experiment, "+".join(EXPERIMENTS[experiment]["train_tasks"]), status, gpu, log.get("step", "n/a"),
            "n/a" if seconds is None else f"{seconds:.0f}", log.get("speed", "n/a"), log.get("memory", "n/a")]


def training_summary(cfg: dict, runs_dir: Path) -> list:
    return [_training_row(e, runs_dir / e) for e, d in EXPERIMENTS.items() if d["train_tasks"]]


def _read_training_log(run_dir: Path) -> dict:
    path = run_dir / "logging.jsonl"
    if not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        if "train_speed(s/it)" in row:
            out["speed"] = row["train_speed(s/it)"]
        if "memory(GiB)" in row:
            out["memory"] = row["memory(GiB)"]
        if "global_step/max_steps" in row:
            out["step"] = row["global_step/max_steps"]
    return out


def build_report(cfg: dict) -> dict:
    runs, warnings = load_runs(cfg)
    reports, seed = resolve(cfg, "reports_dir"), cfg["seed"]
    table = {row: row_metrics(row, runs) for row in ROWS}

    baseline = {t: table["E0 pretrained"]["per_task"][t]["accuracy"] for t in table["E0 pretrained"]["per_task"]}
    single = {}
    for task in TASKS:
        experiment = f"e1_{task.lower()}"
        if experiment in runs:
            single[task] = {t: runs[experiment]["metrics"]["per_task"][t]["accuracy"] for t in TASKS}
    joint = {t: m["accuracy"] for t, m in table["E2 joint"]["per_task"].items()}
    matrix, interference = directed_transfer(baseline, single), joint_interference(joint, single)

    write_results_csv(reports / "results.csv", table)
    write_transfer_csv(reports / "transfer_matrix.csv", matrix)
    plot_transfer_heatmap(matrix, reports / "transfer_heatmap.png")

    e3, e4 = row_tasks("E3 question-type", runs), row_tasks("E4 operation grouping", runs)
    e2, e1 = row_tasks("E2 joint", runs), row_tasks("E1 single-task", runs)
    e4_vs_e3 = macro_paired_ci(e4, e3, TASKS, seed)
    invalids = [table[r]["invalid_rate"] for r in ("E3 question-type", "E4 operation grouping")
                if table[r]["invalid_rate"] is not None]
    decision, rationale = research_decision(e4_vs_e3, max(invalids) if invalids else None)

    summary = {"completed": sorted(runs), "warnings": warnings, "decision": decision, "rationale": rationale,
               "transfer": matrix, "interference": interference, "e4_minus_e3_macro": e4_vs_e3,
               "e4_minus_e3_query_macro": macro_paired_ci(e4, e3, QUERY_TASKS, seed),
               "e4_minus_e3_verify_macro": macro_paired_ci(e4, e3, VERIFY_TASKS, seed)}
    write_json(reports / "summary.json", summary)

    text = render_markdown(cfg, runs, warnings, table, matrix, interference, baseline, e1, e2, e3, e4, summary)
    (reports / "report.md").write_text(text)
    return summary


def render_markdown(cfg, runs, warnings, table, matrix, interference, baseline, e1, e2, e3, e4, summary) -> str:
    seed, train = cfg["seed"], cfg["train"]
    runs_dir, processed = resolve(cfg, "runs_dir"), resolve(cfg, "processed_dir")
    manifest = None
    if (processed / "manifest.json").exists():
        with open(processed / "manifest.json") as fh:
            manifest = json.load(fh)
    by_gpu: dict[str, list[str]] = {}  # evaluations may run on different GPUs (run_queue.sh --gpus)
    for name, r in sorted(runs.items()):
        if g := r["metrics"].get("environment", {}).get("gpu"):
            by_gpu.setdefault(g, []).append(name)
    if len(by_gpu) == 1:
        gpu = next(iter(by_gpu))
    else:
        gpu = "; ".join(f"{g} (eval: {', '.join(names)})" for g, names in by_gpu.items())
    missing = [e for e in EXPERIMENTS if e not in runs]

    md = ["# Multi-task VQA pilot report", ""]
    if not runs:
        md += ["> **Template.** No completed experiments were found, so this report contains no scores.", ""]
    md += ["## A. Experimental setup", "",
           f"- **Model:** `{cfg['model']}` (`{cfg['model_path']}`), 4-bit NF4 base for E0 and all adapters.",
           "- **Dataset:** GQA v1.2 balanced questions. Train pool and internal validation come from the official "
           "train split, and the held-out pilot test comes from the official val split. "
           "**This is not a result on the official GQA test benchmark.**",
           "- **Tasks:** T1 query/no relate, T2 query/relate, T3 verify/no relate, T4 verify/relate, using real "
           "`semantic[].operation` steps. This is a coarse diagnostic, not a complete measure of reasoning similarity.",
           f"- **Training:** LoRA rank {train['lora_rank']}, alpha {train['lora_alpha']}, lr {train['learning_rate']}, "
           f"{train['examples_per_task']} examples per task, effective batch {train['effective_batch_size']} "
           f"(micro-batch {train['micro_batch_size']}), frozen vision encoder and aligner, non-thinking.",
           f"- **Random seed:** {seed} (single seed).",
           f"- **GPU:** {gpu or 'not recorded (no completed evaluation)'}.",
           f"- **Completed evaluations:** {', '.join(sorted(runs)) or 'none'}.",
           f"- **Not completed:** {', '.join(missing) or 'none'}.", ""]
    if manifest:
        counts = [[s, c["questions"], c["unique_images"], c["per_task"]["T1"], c["per_task"]["T2"], c["per_task"]["T3"],
                   c["per_task"]["T4"]] for s, c in manifest["counts"].items()]
        md += ["Data counts:", ""] + _table(["Split", "Questions", "Images", "T1", "T2", "T3", "T4"], counts) + [""]
    else:
        md += ["Data counts: dataset not built yet.", ""]
    md += ["Training runs:", ""] + _table(
        ["Run", "Train tasks", "Status", "GPU", "Step", "Wall s", "s/step", "Peak GiB"], training_summary(cfg, runs_dir)) + [""]
    md += ["Budget note: every task contributes the same examples, so one-, two- and four-task adapters take different "
           "numbers of optimizer steps by design (at defaults 50, 100 and 200).", ""]
    if warnings:
        md += ["**Warnings:**", ""] + [f"- {w}" for w in warnings] + [""]

    md += ["## B. Main results", "",
           "Exact-match accuracy on the held-out pilot test. E1 uses each target task's own single-task adapter. "
           "E3/E4 use oracle routing from dataset metadata.", ""]
    rows = []
    for row, m in table.items():
        rows.append([row, *[_fmt(m["per_task"].get(t, {}).get("accuracy")) for t in TASKS], _fmt(m["macro_accuracy"]),
                     _fmt(m["pooled_accuracy"]), _fmt(m["invalid_rate"])])
    md += _table(["Experiment", *TASKS, "Macro Avg", "Pooled", "Invalid rate"], rows) + ["",
                 "Query tasks (T1, T2) are open-answer; verify tasks (T3, T4) are binary. Prefer per-task and macro "
                 "values, because pooled accuracy can be distorted by answer format.", ""]

    md += ["## C. Directed transfer", "", "T[i→j] = Acc_j(M_i) − Acc_j(M_0). Rows are source tasks and columns are "
           "target tasks. This is source-only cross-task evaluation, **not** joint MTL interference.", ""]
    md += _table(["source \\ target", *TASKS], [[s, *[_fmt(matrix[s][t], signed=True) for t in TASKS]] for s in TASKS])
    md += ["", "![Directed transfer heatmap](transfer_heatmap.png)", "",
           "Positive means the adapter improved the target task over the pretrained model; negative means it degraded "
           "it. Asymmetry means T[i→j] differs from T[j→i].", "",
           "Mean off-diagonal transfer by task relationship (descriptive, four pairs each):", ""]
    md += _table(["Relationship", "Mean transfer", "Pairs available"],
                 [[k, _fmt(v["mean"], signed=True), v["pairs"]] for k, v in transfer_by_relation(matrix).items()])
    md += ["", "Same-operation pairs always cross answer format and same-type pairs never do, so this comparison cannot "
           "separate reasoning similarity from answer format.", ""]
    format_rows = []
    for source in TASKS:
        experiment = f"e1_{source.lower()}"
        cells = [_fmt(runs[experiment]["metrics"]["per_task"][t]["yes_no_prediction_rate"]) if experiment in runs
                 else "not run" for t in TASKS]
        format_rows.append([source, *cells])
    md += ["Share of predictions that are yes/no, per E1 adapter. High values on query targets indicate an answer-format "
           "shift rather than a reasoning effect:", ""]
    md += _table(["source \\ target", *TASKS], format_rows) + [""]

    md += ["## D. MTL interference", "", "I_i = Acc_i(M_joint) − Acc_i(M_i), with a paired bootstrap 95% CI over test "
           "questions. Positive: joint training helped task i. Negative: it hurt task i. E2 is naive joint training, "
           "used as a baseline and not a new method.", ""]
    md += _table(["Task", "E2 joint", "E1 single-task", "I_i [95% CI]"], [
        [t, _fmt(table["E2 joint"]["per_task"].get(t, {}).get("accuracy")),
         _fmt(table["E1 single-task"]["per_task"].get(t, {}).get("accuracy")),
         _fmt_ci(paired_ci(e2[t], e1[t], seed) if t in e2 and t in e1 else None)] for t in TASKS])
    md += ["", "Directed transfer does not necessarily predict joint interference; the two measure different training "
           "conditions.", ""]

    md += ["## E. Preliminary idea assessment", "", "E4 − E3 with paired bootstrap 95% CIs. Both use two adapters and "
           "corresponding budgets, making this the closer comparison. Compared with E2, both have more total adapter "
           "capacity, so any gain over E2 may reflect specialization or capacity rather than grouping alone.", ""]
    effect_rows = [[t, _fmt(table["E3 question-type"]["per_task"].get(t, {}).get("accuracy")),
                    _fmt(table["E4 operation grouping"]["per_task"].get(t, {}).get("accuracy")),
                    _fmt_ci(paired_ci(e4[t], e3[t], seed) if t in e3 and t in e4 else None)] for t in TASKS]
    effect_rows += [["Macro (T1-T4)", _fmt(table["E3 question-type"]["macro_accuracy"]),
                     _fmt(table["E4 operation grouping"]["macro_accuracy"]), _fmt_ci(summary["e4_minus_e3_macro"])],
                    ["Query macro (T1, T2)", _fmt(table["E3 question-type"]["query_macro_accuracy"]),
                     _fmt(table["E4 operation grouping"]["query_macro_accuracy"]),
                     _fmt_ci(summary["e4_minus_e3_query_macro"])],
                    ["Verify macro (T3, T4)", _fmt(table["E3 question-type"]["verify_macro_accuracy"]),
                     _fmt(table["E4 operation grouping"]["verify_macro_accuracy"]),
                     _fmt_ci(summary["e4_minus_e3_verify_macro"])]]
    md += _table(["Target", "E3", "E4", "E4 − E3 [95% CI]"], effect_rows)
    md += ["", "E3 adapters each see one answer format, while E4 adapters must learn both. The E3 versus E4 comparison "
           "therefore also tests answer-format specialization. One small run cannot establish that any grouping or "
           "architecture is superior, and E4 does not validate the general reasoning-operation hypothesis.", ""]

    md += ["## F. Limitations", "",
           "- **Question type vs answer format:** `query`/`verify` is confounded with open vs yes/no answers.",
           "- **Small number of tasks:** four groups support feasibility, not a general taxonomy claim.",
           "- **Limited training:** few optimizer steps, and step counts differ by number of tasks.",
           "- **Single checkpoint** and **single random seed:** CIs cover test-question sampling only, not training "
           "variation.",
           "- **Coarse grouping:** presence of `relate` is one operation among many, and it correlates with program length.",
           "- **Oracle routing:** E3/E4 route by dataset metadata and are not automatic routing systems.",
           "- **Transfer vs MTL:** source-only transfer (E1) and joint training (E2) are different conditions.",
           "- **Adapter capacity:** E3/E4 use two adapters and E2 uses one.",
           "- **Quantized base:** all results use the 4-bit model, which may differ from full-precision accuracy.",
           "- **No benchmark reproduction:** this is a controlled pilot motivated by prior work, not a reproduction of "
           "VQACL, the official GQA benchmark or any other paper.", ""]

    md += ["## G. Research decision", "", f"**{summary['decision']}.** {summary['rationale']}", "",
           "Rule fixed before running: Promising if the E4 − E3 macro CI is above zero, Not supported if below zero, "
           f"otherwise Inconclusive; Inconclusive if invalid generations exceed {INVALID_RATE_LIMIT:.0%}. This is a "
           "mechanical output that should be reviewed against sections C to F. A positive result would justify larger "
           "experiments only.", ""]
    return "\n".join(md)


# --- multi-seed report (configs with train_seeds, e.g. v2_1000train_500test) ------------

SEED_ROWS = ("E1 single-task", "E2 joint", "E3 question-type", "E4 operation grouping")
E4_VS_E3_TARGETS = {"T1": ["T1"], "T2": ["T2"], "T3": ["T3"], "T4": ["T4"], "Macro (T1-T4)": list(TASKS),
                    "Query macro (T1, T2)": list(QUERY_TASKS), "Verify macro (T3, T4)": list(VERIFY_TASKS)}


def contrast_targets(cfg: dict) -> dict:
    if cfg.get("comparison_design") == "crossed_detailed":
        return {"T1": ["T1"], "T2": ["T2"], "T3": ["T3"], "T4": ["T4"],
                "Macro (T1-T4)": list(TASKS),
                "directOf macro (T1, T2)": list(QUERY_TASKS),
                "directWhich macro (T3, T4)": list(VERIFY_TASKS)}
    return E4_VS_E3_TARGETS


def load_run_sets(cfg: dict):
    """Ordered run sets on this config's test: pretrained E0, previous-version adapters, one set per seed."""
    runs_dir, expected, warnings = resolve(cfg, "runs_dir"), cfg["evaluation"]["samples_per_task"], []
    trainable = [e for e, d in EXPERIMENTS.items() if d["train_tasks"]]
    sets = {"pretrained": {}}
    e0 = load_evaluation(runs_dir / "e0", "e0", expected, warnings)
    if e0:
        sets["pretrained"]["e0"] = e0
    for label in cfg.get("previous_versions") or {}:
        sets[label] = {e: r for e in trainable if (r := load_evaluation(runs_dir / label / e, e, expected, warnings))}
    for seed in cfg["train_seeds"]:
        sets[f"seed{seed}"] = {e: r for e in trainable
                               if (r := load_evaluation(run_dir(cfg, e, seed), e, expected, warnings))}
    loaded = [r for runs in sets.values() for r in runs.values()]
    if len({r["metrics"].get("metadata_sha256") for r in loaded}) > 1:
        warnings.append("Evaluations used different test metadata (hash mismatch); comparisons are not valid.")
    return sets, warnings


def _correctness(row_rows: dict) -> dict:
    return {t: {q: float(r["correct"]) for q, r in rows.items()} for t, rows in row_rows.items()}


def _average(items: list) -> dict:
    """Per-question mean correctness over seeds; only tasks present with identical questions in every item."""
    if not items:
        return {}
    tasks = [t for t in items[0] if all(t in i and set(i[t]) == set(items[0][t]) for i in items)]
    return {t: {q: float(np.mean([i[t][q] for i in items])) for q in items[0][t]} for t in tasks}


def _row_values(m: dict) -> list:
    return [m["per_task"].get(t, {}).get("accuracy") for t in TASKS] + [m["macro_accuracy"], m["invalid_rate"]]


def _mean_sd(values: list) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "not run"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


def build_seed_report(cfg: dict) -> dict:
    sets, warnings = load_run_sets(cfg)
    reports, boot_seed, seeds = resolve(cfg, "reports_dir"), cfg["seed"], cfg["train_seeds"]
    images = {q: r["image_id"] for runs in sets.values() for run in runs.values()
              for rows in run["by_task"].values() for q, r in rows.items()}

    # accuracy rows: (run set, experiment row) -> metrics, only rows with at least one task
    accuracy = {}
    if "e0" in sets["pretrained"]:
        accuracy[("pretrained", "E0 pretrained")] = row_metrics("E0 pretrained", sets["pretrained"])
    for name, runs in sets.items():
        if name == "pretrained":
            continue
        for row in SEED_ROWS:
            m = row_metrics(row, runs)
            if m["per_task"]:
                accuracy[(name, row)] = m

    # E4 - E3 per run set, and pooled over seeds (per-question mean correctness)
    contrasts, per_seed = {}, {}
    targets = contrast_targets(cfg)
    seed_sets = [f"seed{s}" for s in seeds]
    e3s, e4s = {}, {}
    for name, runs in sets.items():
        e3, e4 = _correctness(row_tasks("E3 question-type", runs)), _correctness(row_tasks("E4 operation grouping", runs))
        if all(t in e3 and t in e4 for t in TASKS):
            contrasts[name] = {k: cluster_paired_ci(e4, e3, images, ts, boot_seed) for k, ts in targets.items()}
            if name in seed_sets:
                e3s[name], e4s[name] = e3, e4
                per_seed[name] = contrasts[name]["Macro (T1-T4)"]["diff"]
    if e3s:
        pooled_name = f"seeds pooled ({len(e3s)}/{len(seeds)})"
        e3p, e4p = _average(list(e3s.values())), _average(list(e4s.values()))
        contrasts[pooled_name] = {k: cluster_paired_ci(e4p, e3p, images, ts, boot_seed)
                                  for k, ts in targets.items()}

    # Directed transfer T[i->j] = Acc_j(E1_i) - Acc_j(E0), averaged over seeds; and MTL interference
    # I_i = Acc_i(E2) - Acc_i(E1_i), with a seed-pooled, image-clustered CI.
    baseline = {t: m["accuracy"] for t, m in accuracy.get(("pretrained", "E0 pretrained"), {"per_task": {}})["per_task"].items()}
    matrices, e1s, e2s = [], [], []
    for name in seed_sets:
        runs = sets[name]
        single = {t: {u: runs[f"e1_{t.lower()}"]["metrics"]["per_task"][u]["accuracy"] for u in TASKS}
                  for t in TASKS if f"e1_{t.lower()}" in runs}
        if len(single) == len(TASKS) and baseline:
            matrices.append(directed_transfer(baseline, single))
        e1, e2 = _correctness(row_tasks("E1 single-task", runs)), _correctness(row_tasks("E2 joint", runs))
        if all(t in e1 and t in e2 for t in TASKS):
            e1s.append(e1)
            e2s.append(e2)
    transfer = {s_: {t: float(np.mean([m[s_][t] for m in matrices])) for t in TASKS} for s_ in TASKS} if matrices else None
    interference = None
    if e1s:
        e1p, e2p = _average(e1s), _average(e2s)
        interference = {k: cluster_paired_ci(e2p, e1p, images, ts, boot_seed)
                        for k, ts in targets.items() if k in (*TASKS, "Macro (T1-T4)")}

    if len(e3s) == len(seeds):
        invalids = [accuracy[(n, r)]["invalid_rate"] for n in seed_sets for r in ("E3 question-type",
                    "E4 operation grouping") if (n, r) in accuracy]
        decision, rationale = research_decision(contrasts[pooled_name]["Macro (T1-T4)"], max(invalids))
        if cfg.get("comparison_design") == "crossed_detailed" and len(seeds) == 1:
            decision = "Pilot only"
            rationale = ("One training seed cannot establish training stability. " + rationale)
    else:
        decision, rationale = "Pending", (f"E3 and E4 are complete for {len(e3s)} of {len(seeds)} training seeds; "
                                          "the decision uses all seeds.")

    summary = {"version": cfg.get("version"), "train_seeds": seeds, "decision": decision, "rationale": rationale,
               "warnings": warnings, "completed": {n: sorted(r) for n, r in sets.items()},
               "accuracy": {f"{n} | {r}": m for (n, r), m in accuracy.items()},
               "e4_minus_e3": contrasts, "e4_minus_e3_macro_per_seed": per_seed,
               "transfer_mean_over_seeds": transfer, "transfer_seeds": len(matrices),
               "interference_e2_minus_e1": interference, "interference_seeds": len(e1s)}
    write_json(reports / "summary.json", summary)
    reports.mkdir(parents=True, exist_ok=True)
    with open(reports / "results.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["run_set", "experiment", *TASKS, "macro_avg", "query_macro", "verify_macro", "invalid_rate"])
        for (name, row), m in accuracy.items():
            values = [m["per_task"].get(t, {}).get("accuracy") for t in TASKS] + [
                m["macro_accuracy"], m["query_macro_accuracy"], m["verify_macro_accuracy"], m["invalid_rate"]]
            writer.writerow([name, row, *["" if v is None else v for v in values]])
    (reports / "report.md").write_text(render_seed_markdown(cfg, sets, accuracy, contrasts, per_seed, summary, images))
    return summary


def render_seed_markdown(cfg, sets, accuracy, contrasts, per_seed, summary, images) -> str:
    train, seeds = cfg["train"], cfg["train_seeds"]
    manifest_path = resolve(cfg, "processed_dir") / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    md = [f"# Multi-task VQA pilot: {cfg.get('version') or 'multi-seed'}", ""]
    md += ["## Setup", "",
           f"- **Model:** `{cfg['model']}`, 4-bit NF4 QLoRA; LoRA rank {train['lora_rank']}, alpha {train['lora_alpha']}, "
           f"lr {train['learning_rate']}, effective batch {train['effective_batch_size']}, warmup {train['warmup_ratio']}. "
           "Same settings for every experiment and seed.",
           f"- **Data:** {train['examples_per_task']} train examples per task (one pass), "
           f"{cfg['data']['targets']['internal_val']} internal-validation and {cfg['evaluation']['samples_per_task']} "
           f"test questions per task. Dataset seed {cfg['seed']}; training seeds {', '.join(map(str, seeds))}.",
           "- **Equal exposure:** every task contributes the same examples once in every experiment, so E3/E4 "
           "adapters take half the optimizer steps of E2.",
           "- **Test:** GQA balanced val split, excluding every question and image of earlier versions. It is not the "
           "official GQA test benchmark and was not used to choose any hyperparameter.",
           "- **Routing:** E3/E4 use oracle routing from dataset metadata.", ""]
    if cfg.get("comparison_design") == "crossed_detailed":
        md += ["- **Task design:** T1/T2 are `directOf` without/with `relate`; T3/T4 are `directWhich` "
               "without/with `relate`. All four have open answers. Legacy adapter IDs `e3_query`/`e3_verify` "
               "mean the two detailed-type groups in this run.",
               "- **Inference limit:** with one training seed, CIs cover test-image sampling only; this is a pilot.", ""]
    if manifest:
        counts = [[s, c["questions"], c["unique_images"], *[c["per_task"][t] for t in TASKS]]
                  for s, c in manifest["counts"].items()]
        md += _table(["Split", "Questions", "Images", *TASKS], counts) + [""]
        for label, e in (manifest.get("excluded_from_test") or {}).items():
            md += [f"Excluded from the test pool ({label}): {e['questions']} questions and {e['images']} images, "
                   f"including its {e['previous_test_questions']} previously evaluated test questions.", ""]
    if images:
        per_image = np.bincount(np.unique(list(images.values()), return_inverse=True)[1])
        md += [f"Test questions share images: {len(per_image)} images, up to {per_image.max()} questions each, "
               f"{(per_image > 1).mean():.1%} of images with more than one. CIs therefore use a paired bootstrap over "
               "images (all questions of a resampled image move together), "
               f"{BOOTSTRAP_SAMPLES} resamples.", ""]
    rows = [[f"seed{s}", *_training_row(e, run_dir(cfg, e, s))] for s in seeds
            for e in ("e2_joint", "e3_query", "e3_verify", "e4_no_relate", "e4_relate")]
    md += ["Training runs:", ""] + _table(["Seed", "Run", "Tasks", "Status", "GPU", "Step", "Wall s", "s/step",
                                           "Peak GiB"], rows) + [""]
    if summary["warnings"]:
        md += ["**Warnings:**", ""] + [f"- {w}" for w in summary["warnings"]] + [""]

    md += ["## Accuracy", "", "Exact-match accuracy on the new test. Rows from earlier versions use their existing "
           "adapters without retraining.", ""]
    acc_rows = [[name, row, *[_fmt(v) for v in _row_values(m)]] for (name, row), m in accuracy.items()]
    seed_sets = [f"seed{s}" for s in seeds]
    for row in SEED_ROWS:
        ms = [accuracy[(n, row)] for n in seed_sets if (n, row) in accuracy]
        if len(ms) > 1:
            acc_rows.append([f"mean ± sd ({len(ms)} seeds)", row,
                             *[_mean_sd([m["per_task"].get(t, {}).get("accuracy") for m in ms]) for t in TASKS],
                             _mean_sd([m["macro_accuracy"] for m in ms]), _mean_sd([m["invalid_rate"] for m in ms])])
    md += _table(["Run set", "Experiment", *TASKS, "Macro", "Invalid"], acc_rows or [["none", "not run", *[""] * 6]])
    md += [""]

    md += ["## Directed transfer (E1 vs E0)", ""]
    if summary["transfer_mean_over_seeds"]:
        matrix = summary["transfer_mean_over_seeds"]
        md += [f"T[i→j] = Acc_j(E1 adapter trained on i) − Acc_j(E0), mean over {summary['transfer_seeds']} seeds. Rows "
               "are source tasks, columns target tasks.", ""]
        md += _table(["source \\ target", *TASKS], [[s_, *[_fmt(matrix[s_][t], signed=True) for t in TASKS]] for s_ in TASKS])
        md += ["", "Mean off-diagonal transfer by relationship:", ""]
        md += _table(["Relationship", "Mean transfer"],
                     [[k, _fmt(v["mean"], signed=True)] for k, v in transfer_by_relation(matrix).items()])
        if cfg.get("comparison_design") != "crossed_detailed":
            md += ["", "Same-operation pairs always cross answer format, so this cannot separate reasoning from format.", ""]
    else:
        md += ["Not available: needs E0 and all four E1 adapters for at least one seed.", ""]

    md += ["## MTL interference (E2 − E1)", ""]
    if summary["interference_e2_minus_e1"]:
        md += [f"I_i = Acc_i(E2 joint) − Acc_i(E1 single-task on i), seed-pooled over {summary['interference_seeds']} "
               "seeds, image-clustered 95% CI. Positive: joint training helped.", ""]
        md += _table(["Target", "I [95% CI]"], [[k, _fmt_ci(v)] for k, v in summary["interference_e2_minus_e1"].items()])
        md += [""]
    else:
        md += ["Not available: needs E1 and E2 for at least one seed.", ""]

    md += ["## E4 − E3", "", "Paired difference on identical test questions, with image-clustered bootstrap 95% CIs. "
           "The pooled row averages each question's correctness over seeds first; its CI covers test sampling, and the "
           "seed spread below shows training variation.", ""]
    keys = list(contrast_targets(cfg))
    md += _table(["Run set", *keys], [[name, *[_fmt_ci(c[k]) for k in keys]] for name, c in contrasts.items()]
                 or [["none", *["not run"] * len(keys)]])
    if per_seed:
        values = list(per_seed.values())
        spread = f"mean {np.mean(values):+.3f}" + (f", sd {np.std(values, ddof=1):.3f}" if len(values) > 1 else "")
        md += ["", f"Per-seed macro E4 − E3: {', '.join(f'{n} {v:+.3f}' for n, v in per_seed.items())} ({spread}; "
               f"{sum(v > 0 for v in values)}/{len(values)} positive)."]
    if cfg.get("comparison_design") == "crossed_detailed":
        md += ["", "All adapters see open answers; answer-frequency and program-length differences remain possible confounds.", ""]
    else:
        md += ["", "E3 adapters each see one answer format while E4 adapters see both, so this contrast also measures "
               "answer-format specialization.", ""]

    md += ["## Decision", "", f"**{summary['decision']}.** {summary['rationale']}", "",
           "Pre-specified rule, applied to the seed-pooled macro CI: Promising if above zero, Not supported if below "
           f"zero, otherwise Inconclusive; Inconclusive if invalid generations exceed {INVALID_RATE_LIMIT:.0%}.", ""]
    return "\n".join(md)


# --- budget estimate ------------------------------------------------------------------

def estimate_budget(cfg: dict, sec_per_step: float, sec_per_example: float, load_seconds: float) -> list[dict]:
    """Projected duration per stage, in priority order, from measured timings supplied by the user."""
    train, per_task = cfg["train"], cfg["evaluation"]["samples_per_task"]
    val_per_task = cfg["data"]["targets"]["internal_val"] if train["eval_at_end"] else 0
    stages = {"E0": ["e0"], "E1": ["e1_t1", "e1_t2", "e1_t3", "e1_t4"], "E2": ["e2_joint"],
              "E3": ["e3_query", "e3_verify"], "E4": ["e4_no_relate", "e4_relate"]}
    repeats = len(cfg.get("train_seeds") or [cfg["seed"]])  # every trained stage runs once per training seed
    rows, cumulative = [], 0.0
    for stage, experiments in stages.items():
        steps = examples = validation = loads = 0
        for experiment in experiments:
            definition = EXPERIMENTS[experiment]
            n_train = len(definition["train_tasks"]) * train["examples_per_task"]
            steps += max(1, n_train // train["effective_batch_size"]) if n_train else 0
            validation += len(definition["train_tasks"]) * val_per_task
            examples += len(definition["eval_tasks"]) * per_task
            loads += 2 if n_train else 1
        if stage != "E0":
            steps, examples, validation, loads = steps * repeats, examples * repeats, validation * repeats, loads * repeats
        hours = (steps * sec_per_step + (examples + validation) * sec_per_example + loads * load_seconds) / 3600
        cumulative += hours
        rows.append({"stage": stage, "priority": "mandatory" if stage in ("E0", "E1", "E2") else "exploratory",
                     "optimizer_steps": steps, "test_examples": examples, "validation_examples": validation,
                     "hours": hours, "cumulative_hours": cumulative})
    return rows


def _measured_timings(cfg: dict) -> dict:
    """Timings from the smoke runs (this version's, else a previous version's). Nothing is assumed otherwise."""
    candidates = [resolve(cfg, "runs_dir")] + [resolve_path(cfg, p["runs_dir"])
                                              for p in (cfg.get("previous_versions") or {}).values()]
    smoke = next((c / "smoke" for c in candidates if (c / "smoke" / "e0" / "metrics.json").exists()),
                 candidates[0] / "smoke")
    found = {}
    speed = _read_training_log(smoke / "e1_t1").get("speed")
    if speed:
        found["sec_per_step"] = float(speed)
    metrics_path = smoke / "e0" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as fh:
            runtime = json.load(fh).get("runtime", {})
        if runtime.get("seconds_per_example"):
            found["sec_per_example"] = runtime["seconds_per_example"]
        if runtime.get("load_seconds"):
            found["load_seconds"] = runtime["load_seconds"]
    return found


def cmd_estimate(cfg: dict, args) -> int:
    measured = _measured_timings(cfg)
    sec_per_step = args.sec_per_step or measured.get("sec_per_step")
    sec_per_example = args.sec_per_example or measured.get("sec_per_example")
    load_seconds = args.load_seconds if args.load_seconds is not None else measured.get("load_seconds", 0.0)
    if not sec_per_step or not sec_per_example:
        raise SystemExit("No measured timings. Run the smoke tests (README Step 6) or pass --sec-per-step and "
                         "--sec-per-example. No GPU performance is assumed.")
    print(f"Timings: {sec_per_step:.2f} s/step, {sec_per_example:.2f} s/example, {load_seconds:.0f} s/model load")
    print(f"Training seeds: {len(cfg.get('train_seeds') or [cfg['seed']])} (E1-E4 rows include every seed)\n")
    print(f"{'stage':<6}{'priority':<13}{'steps':>7}{'test':>7}{'val':>7}{'hours':>8}{'total':>8}  fits")
    for row in estimate_budget(cfg, sec_per_step, sec_per_example, load_seconds):
        fits = "yes" if row["cumulative_hours"] <= args.hours_available else "NO"
        print(f"{row['stage']:<6}{row['priority']:<13}{row['optimizer_steps']:>7}{row['test_examples']:>7}"
              f"{row['validation_examples']:>7}{row['hours']:>8.2f}{row['cumulative_hours']:>8.2f}  {fits}")
    print("\nA few smoke steps give a rough, noisy estimate; keep a margin. Validation is costed at the generation rate, "
          "which is conservative. If E2 does not fit, reduce train.examples_per_task or evaluation.samples_per_task "
          "for all experiments and rebuild. Finish E0-E2 before starting E3/E4.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config path (default: config.yaml)")
    parser.add_argument("--estimate", action="store_true", help="estimate E0-E4 duration instead of writing the report")
    parser.add_argument("--hours-available", type=float, default=24.0, help="remaining GPU hours (default: 24)")
    parser.add_argument("--sec-per-step", type=float, default=None, help="measured seconds per optimizer step")
    parser.add_argument("--sec-per-example", type=float, default=None, help="measured seconds per evaluation example")
    parser.add_argument("--load-seconds", type=float, default=None, help="measured model load time")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.estimate:
        return cmd_estimate(cfg, args)
    if cfg.get("train_seeds"):
        summary = build_seed_report(cfg)
        print(f"Decision: {summary['decision']}. {summary['rationale']}")
        for warning in summary["warnings"]:
            print(f"WARNING: {warning}")
        print(f"Written to {resolve(cfg, 'reports_dir')}/")
        return 0
    summary = build_report(cfg)
    print(f"Completed: {summary['completed'] or 'none'}")
    print(f"Decision: {summary['decision']}. {summary['rationale']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Written to {resolve(cfg, 'reports_dir')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
