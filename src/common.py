"""Definitions shared by prepare.py, evaluate.py and report.py. Imports no GPU libraries."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TASKS = ("T1", "T2", "T3", "T4")
QUERY_TASKS = ("T1", "T2")   # open-answer
VERIFY_TASKS = ("T3", "T4")  # binary yes/no

# train_tasks: what the adapter sees. eval_tasks: what it is scored on. For E3/E4 this is
# oracle routing from dataset metadata, not an automatic router.
EXPERIMENTS = {
    "e0": {"train_tasks": [], "eval_tasks": list(TASKS)},
    "e1_t1": {"train_tasks": ["T1"], "eval_tasks": list(TASKS)},
    "e1_t2": {"train_tasks": ["T2"], "eval_tasks": list(TASKS)},
    "e1_t3": {"train_tasks": ["T3"], "eval_tasks": list(TASKS)},
    "e1_t4": {"train_tasks": ["T4"], "eval_tasks": list(TASKS)},
    "e2_joint": {"train_tasks": list(TASKS), "eval_tasks": list(TASKS)},
    "e3_query": {"train_tasks": ["T1", "T2"], "eval_tasks": ["T1", "T2"]},
    "e3_verify": {"train_tasks": ["T3", "T4"], "eval_tasks": ["T3", "T4"]},
    "e4_no_relate": {"train_tasks": ["T1", "T3"], "eval_tasks": ["T1", "T3"]},
    "e4_relate": {"train_tasks": ["T2", "T4"], "eval_tasks": ["T2", "T4"]},
}
TRAINABLE = [e for e, d in EXPERIMENTS.items() if d["train_tasks"]]
EXPERIMENT_GROUPS = {
    "e1": ["e1_t1", "e1_t2", "e1_t3", "e1_t4"],
    "e2": ["e2_joint"],
    "e3": ["e3_query", "e3_verify"],
    "e4": ["e4_no_relate", "e4_relate"],
    "all": list(EXPERIMENTS),
}


def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(path.resolve().parent)
    cfg["_config"] = path.name
    if os.environ.get("VQA_GPU"):  # set per step by scripts/run_queue.sh
        cfg["gpu"] = int(os.environ["VQA_GPU"])
    smoke = cfg["train"]["smoke_steps"]
    if not 1 <= smoke <= 5:
        raise ValueError(f"train.smoke_steps must be in 1..5, got {smoke}")
    if not all(isinstance(s, int) for s in cfg.get("train_seeds", [])):
        raise ValueError(f"train_seeds must be a list of integers, got {cfg['train_seeds']}")
    return cfg


def build_hint(cfg: dict) -> str:
    """The build command for this config, for error messages."""
    flag = "" if cfg.get("_config", "config.yaml") == "config.yaml" else f" --config {cfg['_config']}"
    return f"python src/prepare.py{flag} build"


def resolve_path(cfg: dict, value) -> Path:
    """Relative paths are relative to the directory holding the config file (the repository root)."""
    path = Path(value)
    return path if path.is_absolute() else Path(cfg["_root"]) / path


def resolve(cfg: dict, key: str) -> Path:
    """Absolute path for cfg['paths'][key] (or model_path)."""
    return resolve_path(cfg, cfg["model_path"] if key == "model_path" else cfg["paths"][key])


def train_seed(cfg: dict, seed=None) -> int:
    """Training seed (LoRA init and data order). Defaults to the dataset seed; never changes the dataset."""
    return cfg["seed"] if seed is None else int(seed)


def run_dir(cfg: dict, experiment: str, seed=None, smoke: bool = False) -> Path:
    """Where one experiment's adapter and evaluation live.

    runs/<experiment> for the default seed (the v1 layout); runs/seed<N>/<experiment> for any other
    seed, or for every seed when the config sets per_seed_runs. E0 is untrained and has no seed.
    """
    runs = resolve(cfg, "runs_dir")
    if smoke:
        return runs / "smoke" / experiment
    seed = train_seed(cfg, seed)
    if EXPERIMENTS[experiment]["train_tasks"] and (cfg.get("per_seed_runs") or seed != cfg["seed"]):
        return runs / f"seed{seed}" / experiment
    return runs / experiment


def public_config(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def dataset_fingerprint(cfg: dict) -> str:
    """Hash of everything that determines the generated data; used to detect stale builds."""
    relevant = {
        "seed": cfg["seed"],
        "prompt": cfg["prompt_template"],
        "data": cfg["data"],
        "grouping_operation": cfg["grouping_operation"],
        "tasks": cfg["tasks"],
        "examples_per_task": cfg["train"]["examples_per_task"],
        "effective_batch_size": cfg["train"]["effective_batch_size"],
    }
    if cfg.get("previous_versions"):  # their questions/images are excluded from the test; absent in v1
        relevant["previous_versions"] = cfg["previous_versions"]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]


def matching_tasks(question: dict, cfg: dict) -> list[str]:
    """All task groups a question matches. Should have length 0 or 1; audit reports overlaps."""
    types = question.get("types") or {}
    structural = types.get("structural")
    operation = cfg["grouping_operation"]
    has_operation = any(step.get("operation") == operation for step in question.get("semantic") or [])
    return [
        name for name, rule in cfg["tasks"].items()
        if rule["structural"] == structural and rule["has_operation"] == has_operation
        and ("detailed" not in rule or rule["detailed"] == types.get("detailed"))
    ]


def missing_fields(question: dict) -> list[str]:
    missing = [f for f in ("imageId", "question", "answer") if not question.get(f)]
    types = question.get("types") or {}
    missing += [f"types.{f}" for f in ("structural", "semantic", "detailed") if not types.get(f)]
    semantic = question.get("semantic")
    if not semantic:
        missing.append("semantic")
    elif any("operation" not in step for step in semantic):
        missing.append("semantic[].operation")
    return missing


def build_user_content(cfg: dict, question: str) -> str:
    """The single prompt used for training and for every experiment's inference."""
    return cfg["prompt_template"].format(question=question)


# --- answer normalization and scoring -------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize_answer(text) -> str:
    """Deterministic exact-match normalization; applied identically to prediction and answer.

    Lowercases, drops think blocks, removes punctuation (hyphen counts as a separator), drops
    articles and keeps numbers intact. Number words are NOT mapped to digits, and no semantic
    matching is performed.
    """
    s = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.S | re.I)
    s = re.sub(r"<think>.*$", " ", s, flags=re.S | re.I)  # unterminated thinking is not an answer
    s = s.lower().strip()
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)             # 1,000 -> 1000
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x00", s)             # protect decimal point
    s = re.sub(r"(^|\s)-(?=\d)", "\\1\x01", s)            # protect minus sign
    s = re.sub(r"['\u2019]", "", s)
    s = re.sub(r"[^\w\s\x00\x01]|_", " ", s)
    tokens = s.split()
    kept = [t for t in tokens if t not in _ARTICLES] or tokens
    return " ".join(kept).replace("\x00", ".").replace("\x01", "-")


def score_prediction(sample: dict, raw, finish_reason=None) -> dict:
    normalized = normalize_answer(raw)
    invalid = not normalized
    return {
        "question_id": sample["question_id"],
        "image_id": sample["image_id"],
        "task": sample["task"],
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "prediction_raw": raw,
        "prediction_normalized": normalized,
        "correct": (not invalid) and normalized == normalize_answer(sample["answer"]),
        "invalid": invalid,
        "truncated": finish_reason == "length",
    }


# --- metrics --------------------------------------------------------------------------

def wilson_ci(correct: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def compute_metrics(records: list[dict]) -> dict:
    """Per-task accuracy, macro average (equal task weight, only when T1-T4 all exist), pooled."""
    per_task = {}
    for task in TASKS:
        rows = [r for r in records if r["task"] == task]
        if not rows:
            continue
        n, correct = len(rows), sum(r["correct"] for r in rows)
        per_task[task] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "accuracy_ci95": wilson_ci(correct, n),
            "invalid": sum(r["invalid"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows),
            "yes_no_prediction_rate": sum(r["prediction_normalized"] in ("yes", "no") for r in rows) / n,
        }

    def mean_over(tasks):
        if not all(t in per_task for t in tasks):
            return None
        return sum(per_task[t]["accuracy"] for t in tasks) / len(tasks)

    total = len(records)
    return {
        "per_task": per_task,
        "macro_accuracy": mean_over(TASKS),
        "query_macro_accuracy": mean_over(QUERY_TASKS),
        "verify_macro_accuracy": mean_over(VERIFY_TASKS),
        "pooled_accuracy": sum(r["correct"] for r in records) / total if total else None,
        "invalid_rate": sum(r["invalid"] for r in records) / total if total else None,
        "n": total,
    }


# --- small I/O helpers ----------------------------------------------------------------

def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info(gpu=None) -> dict:
    """Package versions and GPU name, recorded next to each run. Never raises."""
    packages = {}
    for name in ("torch", "transformers", "ms-swift", "peft", "bitsandbytes", "accelerate", "qwen-vl-utils"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    info = {"packages": packages, "gpu_index": gpu, "gpu": None}
    try:
        query = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]
        lines = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        info["gpu"] = next((l for l in lines if gpu is None or l.split(",")[0].strip() == str(gpu)), None)
    except (OSError, subprocess.SubprocessError):
        pass
    return info
