#!/usr/bin/env python
"""Audit GQA, build one deterministic dataset, and resolve ms-swift arguments.

Subcommands (all explicit, none runs implicitly):
  audit           stream the annotations and summarize task groups
  build           select samples, extract only the referenced images, write JSONL
  extract-images  (re)extract and validate images for an existing selection
  train-args      print the env/args for one experiment (used by scripts/train.sh)
  run-meta        write the config snapshot / final status of a training run
  check-swift     verify generated flags against the installed ms-swift
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXPERIMENTS, TASKS, TRAINABLE, build_hint, build_user_content, dataset_fingerprint,
                    environment_info, load_config, matching_tasks, missing_fields, public_config, read_jsonl, resolve,
                    resolve_path, run_dir, sha256_file, train_seed, write_json, write_jsonl)

try:
    import ijson
except ImportError:
    ijson = None

SPLITS = ("train_pool", "internal_val", "test")
CONFOUND_NOTE = ("query vs verify is strongly tied to answer format (open vocabulary vs yes/no). Differences "
                 "between these groups may reflect answer format rather than reasoning. Treat as a confound.")


def confound_note(cfg: dict) -> str:
    if cfg.get("comparison_design") == "crossed_detailed":
        return ("All four tasks use open-answer query questions, so answer format is held constant. "
                "Detailed question type, relate operation, answer frequency, and program length may still differ.")
    return CONFOUND_NOTE


class InsufficientData(Exception):
    """Raised instead of silently shrinking or redefining a task group."""


# --- reading annotations --------------------------------------------------------------

@contextmanager
def open_questions(cfg: dict, filename: str):
    """Open a question JSON from data/raw/<file> if extracted, else directly from the ZIP."""
    archive = resolve(cfg, "questions_zip")
    extracted = archive.parent / filename
    if extracted.exists():
        with open(extracted, "rb") as fh:
            yield fh
        return
    if not archive.exists():
        raise SystemExit(f"Neither {extracted} nor {archive} exists. See README Step 2 for the download command.")
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if Path(n).name == filename]
        if not members:
            raise SystemExit(f"{filename} not found in {archive}")
        with zf.open(members[0]) as fh:
            yield fh


def iter_questions(cfg: dict, filename: str):
    """Yield (question_id, question). Streams with ijson; the balanced train file is too large for json.load."""
    with open_questions(cfg, filename) as fh:
        if ijson is not None:
            yield from ijson.kvitems(fh, "")
        else:
            print("WARNING: ijson not installed; loading the whole JSON into memory.", file=sys.stderr)
            yield from json.load(fh).items()


# --- audit ----------------------------------------------------------------------------

def _new_bucket() -> dict:
    return {"n": 0, "images": set(), "structural": Counter(), "semantic": Counter(), "detailed": Counter(),
            "op_steps": Counter(), "op_questions": Counter(), "answers": Counter(), "program_steps": 0}


def _add_to_bucket(bucket: dict, q: dict) -> None:
    ops = [step.get("operation") for step in q.get("semantic") or []]
    types = q.get("types") or {}
    bucket["n"] += 1
    bucket["images"].add(q.get("imageId"))
    bucket["structural"][types.get("structural")] += 1
    bucket["semantic"][types.get("semantic")] += 1
    bucket["detailed"][types.get("detailed")] += 1
    bucket["op_steps"].update(ops)
    bucket["op_questions"].update(set(ops))
    bucket["answers"][q.get("answer")] += 1
    bucket["program_steps"] += len(ops)


def _summarize_bucket(bucket: dict) -> dict:
    n = bucket["n"]
    yes_no = bucket["answers"]["yes"] + bucket["answers"]["no"]
    return {
        "n": n,
        "unique_images": len(bucket["images"]),
        "structural": dict(bucket["structural"]),
        "semantic": dict(bucket["semantic"]),
        "detailed_top10": dict(bucket["detailed"].most_common(10)),
        "operation_step_counts": dict(bucket["op_steps"].most_common()),
        "operation_question_share": {k: v / n for k, v in bucket["op_questions"].most_common()} if n else {},
        "unique_answers": len(bucket["answers"]),
        "answers_top10": dict(bucket["answers"].most_common(10)),
        "yes_no_share": yes_no / n if n else None,
        "yes_share": bucket["answers"]["yes"] / n if n else None,
        "mean_program_length": bucket["program_steps"] / n if n else None,
    }


def audit_split(cfg: dict, filename: str) -> dict:
    buckets = defaultdict(_new_bucket)
    missing, overlaps, unbalanced, total = Counter(), 0, 0, 0
    for _, q in iter_questions(cfg, filename):
        total += 1
        gaps = missing_fields(q)
        missing.update(gaps)
        if q.get("isBalanced") is False:
            unbalanced += 1
        if gaps:
            continue
        matches = matching_tasks(q, cfg)
        overlaps += len(matches) > 1
        _add_to_bucket(buckets[matches[0] if matches else "unassigned"], q)
    return {
        "file": filename,
        "total_questions": total,
        "missing_annotations": dict(missing),
        "not_marked_balanced": unbalanced,
        "questions_matching_multiple_groups": overlaps,
        "groups": {k: _summarize_bucket(buckets[k]) for k in (*TASKS, "unassigned") if k in buckets},
    }


def audit_warnings(audit: dict, cfg: dict) -> list[str]:
    targets = cfg["data"]["targets"]
    need = {"train": targets["train_pool"] + targets["internal_val"], "heldout": targets["test"]}
    warnings = []
    for split, report in audit["splits"].items():
        if report["questions_matching_multiple_groups"]:
            warnings.append(f"{split}: {report['questions_matching_multiple_groups']} questions match several groups; "
                            "the group definitions overlap, revise config.yaml.")
        if report["missing_annotations"]:
            warnings.append(f"{split}: missing annotations {report['missing_annotations']}; those questions are excluded.")
        for task in TASKS:
            n = report["groups"].get(task, {}).get("n", 0)
            if n == 0:
                warnings.append(f"{split}: {task} is EMPTY. Definitions were not changed; revise config.yaml.")
            elif n < need[split]:
                warnings.append(f"{split}: {task} has {n} questions, below the {need[split]} needed; "
                                "lower data.targets evenly across tasks or revise config.yaml.")
    return warnings


def render_audit(audit: dict) -> str:
    lines = ["# GQA audit", "", f"**Confound:** {audit['confound_note']}", ""]
    for split, report in audit["splits"].items():
        lines += [f"## {split}: `{report['file']}`", "",
                  f"Total {report['total_questions']}; missing annotations {report['missing_annotations'] or 'none'}; "
                  f"not marked balanced {report['not_marked_balanced']}; "
                  f"multi-group matches {report['questions_matching_multiple_groups']}.", "",
                  "| Group | Questions | Images | Yes/no answers | Yes share | Unique answers | Mean program length |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for name, g in report["groups"].items():
            fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            lines.append(f"| {name} | {g['n']} | {g['unique_images']} | {fmt(g['yes_no_share'])} | {fmt(g['yes_share'])} "
                         f"| {g['unique_answers']} | {fmt(g['mean_program_length'])} |")
        lines.append("")
        for name, g in report["groups"].items():
            ops = ", ".join(f"{k} {v:.2f}" for k, v in list(g["operation_question_share"].items())[:8])
            lines += [f"**{name}**: structural {g['structural']}; semantic {g['semantic']}", f"  operations: {ops}", ""]
    lines += ["## Warnings", ""] + ([f"- {w}" for w in audit["warnings"]] or ["- none"])
    return "\n".join(lines) + "\n"


def cmd_audit(cfg: dict, _args) -> int:
    data = cfg["data"]
    splits = {"train": audit_split(cfg, data["train_questions_file"]),
              "heldout": audit_split(cfg, data["heldout_questions_file"])}
    audit = {"confound_note": confound_note(cfg), "splits": splits}
    audit["warnings"] = audit_warnings(audit, cfg)
    out = resolve(cfg, "processed_dir")
    write_json(out / "audit.json", audit)
    text = render_audit(audit)
    (out / "audit.md").write_text(text)
    print(text)
    print(f"Written: {out / 'audit.json'}, {out / 'audit.md'}")
    return 0


# --- selection ------------------------------------------------------------------------

def load_eligible(cfg: dict, filename: str) -> list[dict]:
    """Keep compact metadata for questions in exactly one task group, sorted by id for determinism."""
    records, interned = [], {}
    for qid, q in iter_questions(cfg, filename):
        matches = matching_tasks(q, cfg)
        if missing_fields(q) or len(matches) != 1:
            continue
        ops = tuple(step["operation"] for step in q["semantic"])
        records.append({
            "question_id": str(qid),
            "image_id": str(q["imageId"]),
            "question": q["question"],
            "answer": q["answer"],
            "task": matches[0],
            "structural": q["types"]["structural"],
            "semantic_type": q["types"]["semantic"],
            "detailed_type": q["types"]["detailed"],
            "operations": list(interned.setdefault(ops, ops)),
        })
    return sorted(records, key=lambda r: r["question_id"])


def partition_images(image_ids, fraction: float, seed: int) -> set:
    """Images reserved for internal validation; split is by image so questions cannot leak."""
    ordered = sorted(set(image_ids))
    random.Random(f"{seed}:image_partition").shuffle(ordered)
    return set(ordered[:int(len(ordered) * fraction)])


def round_robin_sample(records: list[dict], n: int, seed_key: str) -> list[dict]:
    """Seeded sample preferring distinct images: one question per image, then a second, and so on.

    The order is meaningful: a prefix of the result is itself a valid smaller sample.
    """
    rng = random.Random(seed_key)
    by_image = defaultdict(list)
    for record in records:
        by_image[record["image_id"]].append(record)
    images = sorted(by_image)
    rng.shuffle(images)
    for image in images:
        rng.shuffle(by_image[image])
    picked, depth = [], 0
    while len(picked) < n:
        layer = [by_image[i][depth] for i in images if depth < len(by_image[i])]
        if not layer:
            break
        picked.extend(layer)
        depth += 1
    return picked[:n]


def previous_selection(cfg: dict) -> dict:
    """Question and image ids used by earlier dataset versions (any split), from their metadata.jsonl."""
    previous = {}
    for label, entry in (cfg.get("previous_versions") or {}).items():
        path = resolve_path(cfg, entry["processed_dir"]) / "metadata.jsonl"
        if not path.exists():
            raise SystemExit(f"previous_versions.{label}: {path} not found; cannot guarantee a fresh test set.")
        rows = read_jsonl(path)
        previous[label] = {"question_ids": {r["question_id"] for r in rows}, "image_ids": {r["image_id"] for r in rows},
                           "test_questions": sum(r["split"] == "test" for r in rows),
                           "metadata_sha256": sha256_file(path)}
    return previous


def select_samples(train_records: list[dict], heldout_records: list[dict], cfg: dict,
                   previous: dict | None = None) -> list[dict]:
    """Seeded, balanced selection. The test pool drops every question and image of a previous version."""
    seed, targets = cfg["seed"], cfg["data"]["targets"]
    used_questions = set().union(*(p["question_ids"] for p in (previous or {}).values()))
    used_images = set().union(*(p["image_ids"] for p in (previous or {}).values()))
    val_images = partition_images((r["image_id"] for r in train_records),
                                  cfg["data"]["internal_val_image_fraction"], seed)
    pools = {split: defaultdict(list) for split in SPLITS}
    for r in train_records:
        pools["internal_val" if r["image_id"] in val_images else "train_pool"][r["task"]].append(r)
    for r in heldout_records:
        if r["question_id"] not in used_questions and r["image_id"] not in used_images:
            pools["test"][r["task"]].append(r)

    available = {s: {t: len(pools[s][t]) for t in TASKS} for s in SPLITS}
    print(f"Available after exclusions (split/task): {available}", flush=True)
    short = [f"{s}/{t}: {available[s][t]} < {targets[s]}" for s in SPLITS for t in TASKS if available[s][t] < targets[s]]
    if short:
        balanced = {s: min(available[s].values()) for s in SPLITS}
        raise InsufficientData(
            "Not enough samples; nothing was written and no group was changed.\n  " + "\n  ".join(short)
            + f"\nAvailable per split/task: {available}"
            + f"\nLargest balanced option for data.targets in config.yaml: {balanced}")

    selected = []
    for split in SPLITS:
        for task in TASKS:
            sample = round_robin_sample(pools[split][task], targets[split], f"{seed}:{split}:{task}")
            selected += [{**r, "split": split, "rank": rank} for rank, r in enumerate(sample)]
    validate_selection(selected, used_questions, used_images)
    return selected


def validate_selection(samples: list[dict], used_questions=frozenset(), used_images=frozenset()) -> None:
    """Fail loudly on leakage or duplicates; never repair by dropping samples."""
    ids = [s["question_id"] for s in samples]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    if duplicates:
        raise ValueError(f"Duplicate question ids across splits: {duplicates[:5]}")
    images = {s: {x["image_id"] for x in samples if x["split"] == s} for s in SPLITS}
    if images["train_pool"] & images["internal_val"]:
        raise ValueError("Image leakage between train_pool and internal_val")
    leaked = (images["train_pool"] | images["internal_val"]) & images["test"]
    if leaked:
        raise ValueError(f"Image leakage into the held-out test: {sorted(leaked)[:5]}")
    test = [x for x in samples if x["split"] == "test"]
    reused = {x["question_id"] for x in test} & set(used_questions) or images["test"] & set(used_images)
    if reused:
        raise ValueError(f"Held-out test reuses a previous version's questions or images: {sorted(reused)[:5]}")
    bad_tasks = {s["task"] for s in samples} - set(TASKS)
    if bad_tasks:
        raise ValueError(f"Samples outside T1-T4: {bad_tasks}")
    counts = Counter((s["split"], s["task"]) for s in samples)
    for split in SPLITS:
        if len({counts[(split, t)] for t in TASKS}) != 1:
            raise ValueError(f"Unbalanced tasks in {split}: { {t: counts[(split, t)] for t in TASKS} }")


# --- image extraction -----------------------------------------------------------------

def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def extract_images(cfg: dict, image_ids) -> dict:
    """Extract only the referenced images; existing valid files are kept. Paths never come from the archive."""
    image_ids = sorted(set(image_ids))
    out_dir = resolve(cfg, "images_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"requested": len(image_ids), "extracted": 0, "already_present": 0, "missing": [], "corrupt": []}
    pending = []
    for image_id in image_ids:
        if not re.fullmatch(r"[\w-]+", image_id):
            report["corrupt"].append(image_id)
        elif (out_dir / f"{image_id}.jpg").exists() and is_valid_image(out_dir / f"{image_id}.jpg"):
            report["already_present"] += 1
        else:
            pending.append(image_id)
    if pending:
        archive = resolve(cfg, "images_zip")
        if not archive.exists():
            raise SystemExit(f"{archive} does not exist. See README Step 2 for the download command.")
        with zipfile.ZipFile(archive) as zf:
            members = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
            for image_id in pending:
                member = members.get(f"{image_id}.jpg")
                if member is None:
                    report["missing"].append(image_id)
                    continue
                target = out_dir / f"{image_id}.jpg"
                temp = target.with_suffix(".tmp")
                temp.write_bytes(zf.read(member))
                temp.replace(target)
                if is_valid_image(target):
                    report["extracted"] += 1
                else:
                    report["corrupt"].append(image_id)
    write_json(resolve(cfg, "processed_dir") / "image_report.json", report)
    return report


# --- experiment datasets --------------------------------------------------------------

def to_swift_record(sample: dict, cfg: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": build_user_content(cfg, sample["question"])},
                     {"role": "assistant", "content": sample["answer"]}],
        "images": [sample["image_path"]],
    }


def training_steps(n_examples: int, effective_batch_size: int) -> int:
    """Optimizer steps for one pass without repeating data. A partial final batch is dropped."""
    return max(1, n_examples // effective_batch_size)


def experiment_samples(samples: list[dict], experiment: str, split: str, cfg: dict) -> list[dict]:
    """Per-task budget is a fixed prefix of each pool, so E1-E4 share identical examples per task."""
    tasks = EXPERIMENTS[experiment]["train_tasks"]
    limit = cfg["train"]["examples_per_task"] if split == "train_pool" else None
    chosen = []
    for task in tasks:
        rows = sorted((s for s in samples if s["split"] == split and s["task"] == task), key=lambda s: s["rank"])
        chosen += rows[:limit] if limit is not None else rows
    random.Random(f"{cfg['seed']}:{experiment}:{split}").shuffle(chosen)
    return chosen


def write_experiment_datasets(samples: list[dict], cfg: dict) -> dict:
    out = resolve(cfg, "processed_dir")
    train_cfg = cfg["train"]
    experiments = {}
    for experiment in TRAINABLE:
        train_rows = experiment_samples(samples, experiment, "train_pool", cfg)
        val_rows = experiment_samples(samples, experiment, "internal_val", cfg)
        train_file, val_file = out / "train" / f"{experiment}.jsonl", out / "val" / f"{experiment}.jsonl"
        write_jsonl(train_file, (to_swift_record(s, cfg) for s in train_rows))
        write_jsonl(val_file, (to_swift_record(s, cfg) for s in val_rows))
        experiments[experiment] = {
            "train_tasks": EXPERIMENTS[experiment]["train_tasks"],
            "examples_per_task": train_cfg["examples_per_task"],
            "n_train": len(train_rows),
            "n_internal_val": len(val_rows),
            "max_steps": training_steps(len(train_rows), train_cfg["effective_batch_size"]),
            "train_file": str(train_file),
            "val_file": str(val_file),
            "train_sha256": sha256_file(train_file),
        }
    return experiments


def split_counts(samples: list[dict]) -> dict:
    counts = {}
    for split in SPLITS:
        rows = [s for s in samples if s["split"] == split]
        counts[split] = {"questions": len(rows), "unique_images": len({s["image_id"] for s in rows}),
                         "per_task": {t: sum(s["task"] == t for s in rows) for t in TASKS},
                         "yes_no_share_per_task": {
                             t: (sum(s["answer"] in ("yes", "no") for s in rows if s["task"] == t)
                                 / max(1, sum(s["task"] == t for s in rows))) for t in TASKS}}
    return counts


def cmd_build(cfg: dict, args) -> int:
    train_cfg, data = cfg["train"], cfg["data"]
    if train_cfg["examples_per_task"] > data["targets"]["train_pool"]:
        raise SystemExit("train.examples_per_task exceeds data.targets.train_pool")
    if cfg["evaluation"]["samples_per_task"] > data["targets"]["test"]:
        raise SystemExit("evaluation.samples_per_task exceeds data.targets.test")

    print("Reading annotations (streaming; the train file takes minutes)...", flush=True)
    previous = previous_selection(cfg)
    train_records = load_eligible(cfg, data["train_questions_file"])
    heldout_records = load_eligible(cfg, data["heldout_questions_file"])
    try:
        samples = select_samples(train_records, heldout_records, cfg, previous)
    except InsufficientData as error:
        print(error, file=sys.stderr)
        return 2

    processed, images_dir = resolve(cfg, "processed_dir"), resolve(cfg, "images_dir")
    for sample in samples:
        sample["image_path"] = str(images_dir / f"{sample['image_id']}.jpg")
    write_jsonl(processed / "metadata.jsonl", samples)
    print(f"Selected {len(samples)} questions on {len({s['image_id'] for s in samples})} images.")
    if args.skip_extract:
        print("--skip-extract: stopping before image extraction. Training JSONL was not written.")
        return 0

    report = extract_images(cfg, (s["image_id"] for s in samples))
    print(f"Images: { {k: (len(v) if isinstance(v, list) else v) for k, v in report.items()} }")
    if report["missing"] or report["corrupt"]:
        print(f"Missing or corrupt images; see {processed / 'image_report.json'}. "
              "Training JSONL was not written and no sample was replaced.", file=sys.stderr)
        return 3

    manifest = {
        "fingerprint": dataset_fingerprint(cfg),
        "version": cfg.get("version"),
        "seed": cfg["seed"],
        "protocol": "Train pool and internal validation come from the official balanced train split; the held-out "
                    "pilot test comes from the official balanced val split. This is not the GQA test benchmark.",
        "confound_note": confound_note(cfg),
        "counts": split_counts(samples),
        "excluded_from_test": {label: {"questions": len(p["question_ids"]), "images": len(p["image_ids"]),
                                       "previous_test_questions": p["test_questions"],
                                       "metadata_sha256": p["metadata_sha256"]} for label, p in previous.items()},
        "experiments": write_experiment_datasets(samples, cfg),
        "metadata_sha256": sha256_file(processed / "metadata.jsonl"),
        "config": public_config(cfg),
    }
    write_json(processed / "manifest.json", manifest)
    print(f"Written: {processed}/metadata.jsonl, manifest.json, train/*.jsonl, val/*.jsonl")
    return 0


def cmd_extract_images(cfg: dict, _args) -> int:
    metadata = resolve(cfg, "processed_dir") / "metadata.jsonl"
    if not metadata.exists():
        raise SystemExit(f"{metadata} not found. Run `python src/prepare.py build --skip-extract` first.")
    report = extract_images(cfg, (s["image_id"] for s in read_jsonl(metadata)))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}))
    return 3 if report["missing"] or report["corrupt"] else 0


# --- ms-swift launcher support --------------------------------------------------------

def load_manifest(cfg: dict) -> dict:
    path = resolve(cfg, "processed_dir") / "manifest.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `{build_hint(cfg)}` first.")
    with open(path) as fh:
        manifest = json.load(fh)
    if manifest["fingerprint"] != dataset_fingerprint(cfg):
        raise SystemExit(f"{cfg['_config']} changed since the dataset was built. Re-run `{build_hint(cfg)}`.")
    return manifest


def swift_env(cfg: dict) -> dict:
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(cfg["gpu"]),
        "IMAGE_MAX_TOKEN_NUM": str(cfg["train"]["image_max_token_num"]),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "USE_HF": "1",
    }


def swift_args(cfg: dict, experiment: str, entry: dict, smoke: bool = False, seed=None) -> list[str]:
    """`swift sft` arguments. Every experiment differs only in dataset, output dir, step count and seed."""
    t = cfg["train"]
    accumulation, remainder = divmod(t["effective_batch_size"], t["micro_batch_size"])
    if remainder:
        raise SystemExit("train.effective_batch_size must be a multiple of train.micro_batch_size")
    steps = t["smoke_steps"] if smoke else entry["max_steps"]
    args = {
        "model": str(resolve(cfg, "model_path")),
        "tuner_type": "lora",
        "dataset": entry["train_file"],
        "torch_dtype": t["torch_dtype"],
        "quant_method": t["quant_method"],
        "quant_bits": t["quant_bits"],
        "bnb_4bit_compute_dtype": t["torch_dtype"],
        "bnb_4bit_quant_type": t["bnb_4bit_quant_type"],
        "bnb_4bit_use_double_quant": t["bnb_4bit_use_double_quant"],
        "freeze_vit": t["freeze_vit"],
        "freeze_aligner": t["freeze_aligner"],
        "lora_rank": t["lora_rank"],
        "lora_alpha": t["lora_alpha"],
        "target_modules": t["target_modules"],
        "add_non_thinking_prefix": True,
        "loss_scale": "ignore_empty_think",
        "learning_rate": t["learning_rate"],
        "warmup_ratio": t["warmup_ratio"],
        "max_steps": steps,
        "per_device_train_batch_size": t["micro_batch_size"],
        "per_device_eval_batch_size": t["micro_batch_size"],
        "gradient_accumulation_steps": accumulation,
        "max_length": t["max_length"],
        "logging_steps": 1 if smoke else t["logging_steps"],
        "save_strategy": "steps",
        "save_steps": steps,
        "save_total_limit": 1,
        "save_only_model": True,
        "output_dir": str(run_dir(cfg, experiment, seed, smoke)),
        "add_version": False,
        "report_to": "none",
        "seed": train_seed(cfg, seed),
        "data_seed": train_seed(cfg, seed),
        "dataset_num_proc": 1,
        "dataloader_num_workers": t["dataloader_num_workers"],
    }
    if t["eval_at_end"] and not smoke:
        args.update(val_dataset=entry["val_file"], eval_strategy="steps", eval_steps=steps)
    else:
        args["eval_strategy"] = "no"
    flat = []
    for key, value in args.items():
        flat += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    return flat


def _train_setup(cfg: dict, experiment: str, smoke: bool, seed=None):
    if experiment not in TRAINABLE:
        raise SystemExit(f"Unknown trainable experiment {experiment!r}. Choose from: {', '.join(TRAINABLE)}")
    manifest = load_manifest(cfg)
    entry = manifest["experiments"][experiment]
    return swift_env(cfg), swift_args(cfg, experiment, entry, smoke, seed), entry


def cmd_train_args(cfg: dict, args) -> int:
    """Print KEY=VALUE env lines, a `--` separator, then one argv token per line."""
    env, argv, _ = _train_setup(cfg, args.experiment, args.smoke, args.seed)
    model_path = resolve(cfg, "model_path")
    if not args.dry_run and not model_path.exists():
        raise SystemExit(f"{model_path} not found. See README Step 5; nothing is downloaded implicitly.")
    print("\n".join([f"{k}={v}" for k, v in env.items()] + ["--"] + argv))
    return 0


def cmd_run_meta(cfg: dict, args) -> int:
    path = run_dir(cfg, args.experiment, args.seed, args.smoke) / "run_config.json"
    if args.phase == "start":
        env, argv, entry = _train_setup(cfg, args.experiment, args.smoke, args.seed)
        write_json(path, {"experiment": args.experiment, "smoke": args.smoke, "started_at": time.time(),
                          "train_seed": train_seed(cfg, args.seed), "version": cfg.get("version"),
                          "swift_env": env, "swift_args": argv, "dataset": entry, "config": public_config(cfg),
                          "environment": environment_info(cfg["gpu"]), "status": "running"})
    else:
        with open(path) as fh:
            meta = json.load(fh)
        meta.update(status="finished" if args.exit_code == 0 else "failed", exit_code=args.exit_code,
                    wall_seconds=args.seconds)
        write_json(path, meta)
    return 0


def cmd_check_swift(cfg: dict, _args) -> int:
    """Compare every generated flag against SftArguments in the installed ms-swift."""
    import dataclasses
    from importlib.metadata import version

    from swift.arguments.sft_args import SftArguments  # imported only by this explicit command
    known = {f.name for f in dataclasses.fields(SftArguments)}
    entry = {"train_file": "train.jsonl", "val_file": "val.jsonl", "max_steps": 50}
    flags = {a[2:] for a in swift_args(cfg, "e1_t1", entry) + swift_args(cfg, "e1_t1", entry, smoke=True)
             if a.startswith("--")}
    unknown = sorted(flags - known)
    print(f"ms-swift {version('ms-swift')}: checked {len(flags)} flags; unknown: {unknown or 'none'}")
    return 1 if unknown else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="config path (default: config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="summarize task groups; writes data/processed/audit.{json,md}")
    build = sub.add_parser("build", help="select samples, extract referenced images, write JSONL and manifest")
    build.add_argument("--skip-extract", action="store_true", help="stop after selecting samples")
    sub.add_parser("extract-images", help="(re)extract and validate images for the existing selection")
    train_args = sub.add_parser("train-args", help="print env and swift args for one experiment")
    train_args.add_argument("experiment")
    train_args.add_argument("--smoke", action="store_true")
    train_args.add_argument("--dry-run", action="store_true", help="skip the model-path check")
    train_args.add_argument("--seed", type=int, default=None, help="training seed (default: the dataset seed)")
    run_meta = sub.add_parser("run-meta", help="write the run snapshot (used by scripts/train.sh)")
    run_meta.add_argument("experiment")
    run_meta.add_argument("--phase", choices=["start", "finish"], required=True)
    run_meta.add_argument("--smoke", action="store_true")
    run_meta.add_argument("--exit-code", type=int, default=0)
    run_meta.add_argument("--seconds", type=float, default=None)
    run_meta.add_argument("--seed", type=int, default=None)
    sub.add_parser("check-swift", help="verify generated flags against the installed ms-swift")
    args = parser.parse_args()

    commands = {"audit": cmd_audit, "build": cmd_build, "extract-images": cmd_extract_images,
                "train-args": cmd_train_args, "run-meta": cmd_run_meta, "check-swift": cmd_check_swift}
    return commands[args.command](load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
