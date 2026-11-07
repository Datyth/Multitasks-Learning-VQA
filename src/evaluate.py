#!/usr/bin/env python
"""Evaluate one experiment at a time, sequentially, on one GPU, with deterministic decoding.

  python src/evaluate.py --experiment e0
  python src/evaluate.py --experiment e1_t1
  python src/evaluate.py --experiment e3          # both E3 adapters, oracle-routed
  python src/evaluate.py --experiment all         # e0 plus every adapter that exists
  python src/evaluate.py --experiment e0 --limit 2   # smoke test, written under runs/smoke/
  python src/evaluate.py --config config.v2_1000train_500test.yaml --experiment e3 --seed 43
  python src/evaluate.py --config config.v2_1000train_500test.yaml --experiment e3 --adapters-from v1_400train

An interrupted run resumes from predictions.jsonl. A finished run is skipped unless --overwrite.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EXPERIMENT_GROUPS, EXPERIMENTS, build_hint, build_user_content, compute_metrics,
                    environment_info, load_config, public_config, read_jsonl, resolve, resolve_path, run_dir,
                    score_prediction, sha256_file, train_seed, write_json)


def find_adapter(run_dir: Path, required_steps: int | None = None):
    """Latest checkpoint holding a LoRA adapter, or None. Rejects a run that stopped early."""
    candidates = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "adapter_config.json").exists():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    step, path = max(candidates)
    if required_steps is not None and step != required_steps:
        return None
    return path


def expand_experiments(name: str) -> list[str]:
    if name in EXPERIMENT_GROUPS:
        return EXPERIMENT_GROUPS[name]
    if name in EXPERIMENTS:
        return [name]
    raise SystemExit(f"Unknown experiment {name!r}. Choose from: {', '.join([*EXPERIMENTS, *EXPERIMENT_GROUPS])}")


def select_test_samples(metadata: list[dict], tasks: list[str], per_task: int) -> list[dict]:
    samples = []
    for task in tasks:
        rows = sorted((s for s in metadata if s["split"] == "test" and s["task"] == task), key=lambda s: s["rank"])
        if len(rows) < per_task:
            raise SystemExit(f"{task}: only {len(rows)} test samples, {per_task} requested. Rebuild or lower the budget.")
        samples += rows[:per_task]
    return samples


def run_inference(samples: list[dict], generate, cfg: dict, predictions_path: Path) -> list[dict]:
    """Generate and score sequentially, appending each record immediately.

    `generate(image_path, prompt)` only ever receives the image and the shared prompt, so no
    task label, operation or ground truth can reach the model.
    """
    done = {r["question_id"]: r for r in read_jsonl(predictions_path)} if predictions_path.exists() else {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "a", encoding="utf-8") as fh:
        for index, sample in enumerate(samples, 1):
            if sample["question_id"] in done:
                continue
            raw, finish_reason = generate(sample["image_path"], build_user_content(cfg, sample["question"]))
            record = score_prediction(sample, raw, finish_reason)
            done[sample["question_id"]] = record
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if index % 50 == 0:
                print(f"  {index}/{len(samples)}", flush=True)
    return [done[s["question_id"]] for s in samples]


def load_generator(cfg: dict, adapter: Path | None):
    """Load the same 4-bit base for E0 and every adapter through the ms-swift inference API."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(cfg["train"]["image_max_token_num"])
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["USE_HF"] = "1"

    import torch
    from swift import get_model_processor, get_template
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from transformers import BitsAndBytesConfig

    t = cfg["train"]
    dtype = getattr(torch, t["torch_dtype"])
    quantization = None
    if t["quant_method"] == "bnb":
        quantization = BitsAndBytesConfig(
            load_in_4bit=t["quant_bits"] == 4, load_in_8bit=t["quant_bits"] == 8,
            bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type=t["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=t["bnb_4bit_use_double_quant"])
    elif t["quant_method"]:
        raise SystemExit(f"evaluate.py supports quant_method bnb or null, got {t['quant_method']}")

    model, processor = get_model_processor(str(resolve(cfg, "model_path")), torch_dtype=dtype,
                                           quantization_config=quantization)
    template = get_template(processor, enable_thinking=False)
    engine = TransformersEngine(model, template=template, adapters=[str(adapter)] if adapter else None)
    evaluation = cfg["evaluation"]
    request_config = RequestConfig(max_tokens=evaluation["max_new_tokens"], temperature=evaluation["temperature"])

    def generate(image_path: str, prompt: str):
        request = InferRequest(messages=[{"role": "user", "content": prompt}], images=[image_path])
        choice = engine.infer([request], request_config=request_config, use_tqdm=False)[0].choices[0]
        return choice.message.content, choice.finish_reason

    def release():
        nonlocal engine, model
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        del engine, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        return peak

    return generate, release


def _required_steps(processed_dir: Path, experiment: str) -> int:
    with open(processed_dir / "manifest.json") as fh:
        return json.load(fh)["experiments"][experiment]["max_steps"]


def evaluate_experiment(experiment: str, cfg: dict, metadata_path: Path, limit=None, overwrite=False,
                        dry_run=False, generator_factory=load_generator, seed=None, adapters_from=None) -> str:
    """Returns one of: evaluated, skipped-done, skipped-missing, dry-run.

    adapters_from names an entry of cfg['previous_versions']: its finished adapter is scored on this
    config's test set, and the output goes to runs_dir/<label>/<experiment>. Nothing is retrained.
    """
    smoke = limit is not None
    definition = EXPERIMENTS[experiment]
    if adapters_from:
        if not definition["train_tasks"]:
            print(f"[{experiment}] skipped: --adapters-from applies to trained experiments only")
            return "skipped-missing"
        previous = (cfg.get("previous_versions") or {}).get(adapters_from)
        if previous is None:
            raise SystemExit(f"{adapters_from!r} is not listed under previous_versions in the config")
        adapter_dir = resolve_path(cfg, previous["runs_dir"]) / experiment
        required = _required_steps(resolve_path(cfg, previous["processed_dir"]), experiment)
        out_dir = resolve(cfg, "runs_dir") / ("smoke" if smoke else "") / adapters_from / experiment
    else:
        adapter_dir = out_dir = run_dir(cfg, experiment, seed, smoke)
        required = None if smoke or not definition["train_tasks"] else _required_steps(resolve(cfg, "processed_dir"),
                                                                                         experiment)

    adapter = None
    if definition["train_tasks"]:
        adapter = find_adapter(adapter_dir, required)
        if adapter is None:
            print(f"[{experiment}] skipped: no finished adapter in {adapter_dir}")
            return "skipped-missing"

    per_task = limit if smoke else cfg["evaluation"]["samples_per_task"]
    samples = select_test_samples(read_jsonl(metadata_path), definition["eval_tasks"], per_task)
    metrics_path, predictions_path = out_dir / "metrics.json", out_dir / "predictions.jsonl"
    if dry_run:
        print(f"[{experiment}] adapter={adapter or 'none (pretrained)'} tasks={definition['eval_tasks']} "
              f"samples={len(samples)} output={out_dir}")
        return "dry-run"
    if metrics_path.exists() and not overwrite:
        print(f"[{experiment}] skipped: {metrics_path} exists (use --overwrite)")
        return "skipped-done"
    if overwrite:
        predictions_path.unlink(missing_ok=True)
        metrics_path.unlink(missing_ok=True)

    print(f"[{experiment}] loading (adapter: {adapter or 'none'})", flush=True)
    start = time.time()
    generate, release = generator_factory(cfg, adapter)
    load_seconds = time.time() - start
    start = time.time()
    already_done = len(read_jsonl(predictions_path)) if predictions_path.exists() else 0
    records = run_inference(samples, generate, cfg, predictions_path)
    inference_seconds = time.time() - start
    peak_mib = release()

    metrics = compute_metrics(records)
    new_examples = len(records) - already_done
    metrics.update({
        "experiment": experiment,
        "model": cfg["model"],
        "adapter": str(adapter) if adapter else None,
        "adapter_source": adapters_from,
        "train_seed": None if adapters_from or not definition["train_tasks"] else train_seed(cfg, seed),
        "test_version": cfg.get("version"),
        "eval_tasks": definition["eval_tasks"],
        "complete": not smoke,
        "limit": limit,
        "metadata_sha256": sha256_file(metadata_path),
        "runtime": {"load_seconds": load_seconds, "inference_seconds": inference_seconds,
                    "seconds_per_example": inference_seconds / new_examples if new_examples else None,
                    "peak_gpu_memory_mib": peak_mib},
        "environment": environment_info(cfg["gpu"]),
        "config": public_config(cfg),
    })
    write_json(metrics_path, metrics)
    macro = metrics["macro_accuracy"]
    print(f"[{experiment}] " + ", ".join(f"{t}={m['accuracy']:.3f}" for t, m in metrics["per_task"].items())
          + (f", macro={macro:.3f}" if macro is not None else "")
          + f", invalid={metrics['invalid_rate']:.3f} -> {metrics_path}")
    return "evaluated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True,
                        help=f"one of {', '.join([*EXPERIMENTS, *EXPERIMENT_GROUPS])}")
    parser.add_argument("--config", default=None, help="config path (default: config.yaml)")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: N samples per task, smoke adapter, output under runs/smoke/<id>")
    parser.add_argument("--overwrite", action="store_true", help="discard existing predictions for this experiment")
    parser.add_argument("--dry-run", action="store_true", help="show what would run; loads no model")
    parser.add_argument("--seed", type=int, default=None, help="training seed of the adapter (default: dataset seed)")
    parser.add_argument("--adapters-from", default=None, metavar="LABEL",
                        help="score a previous version's finished adapters (config previous_versions) on this test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_path = resolve(cfg, "processed_dir") / "metadata.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"{metadata_path} not found. Run `{build_hint(cfg)}` first.")
    model_path = resolve(cfg, "model_path")
    if not args.dry_run and not model_path.exists():
        raise SystemExit(f"{model_path} not found. See README Step 5; nothing is downloaded implicitly.")

    outcomes = {e: evaluate_experiment(e, cfg, metadata_path, args.limit, args.overwrite, args.dry_run,
                                       seed=args.seed, adapters_from=args.adapters_from)
                for e in expand_experiments(args.experiment)}
    print("Summary:", outcomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
