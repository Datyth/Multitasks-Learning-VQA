#!/usr/bin/env python
"""Recreate the 25-per-task GQA val holdout excluded from the v3 test.

Requires the local GQA questions ZIP and v1/v2 metadata. Extract images with
--extract-images only when evaluating models on this small holdout.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from common import load_config, read_jsonl, sha256_file, write_json, write_jsonl
from prepare import extract_images, load_eligible

SELECTION_SEED = 20261107
TASKS = ("T1", "T2", "T3", "T4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-images", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.v2_1000train_500test.yaml")
    out = ROOT / "data/quick_holdout_25per_task"
    out.mkdir(parents=True, exist_ok=True)
    cfg["paths"]["processed_dir"] = str(out)
    cfg["paths"]["images_dir"] = str(out / "images")
    previous = [ROOT / "data/processed/metadata.jsonl", ROOT / "data/v2_1000train_500test/metadata.jsonl"]
    for path in previous:
        if not path.exists():
            raise SystemExit(f"Missing {path}; build v1 and v2 before this holdout.")
    old = [row for path in previous for row in read_jsonl(path)]
    used_questions = {row["question_id"] for row in old}
    used_images = {row["image_id"] for row in old}
    eligible = load_eligible(cfg, cfg["data"]["heldout_questions_file"])

    rng = random.Random(SELECTION_SEED)
    chosen, selected_images = [], set()
    for task in TASKS:
        candidates = [row for row in eligible if row["task"] == task
                      and row["question_id"] not in used_questions and row["image_id"] not in used_images]
        rng.shuffle(candidates)
        picked = []
        for row in candidates:
            if row["image_id"] in selected_images:
                continue
            picked.append({**row, "split": "test", "rank": len(picked),
                           "image_path": str(out / "images" / f"{row['image_id']}.jpg")})
            selected_images.add(row["image_id"])
            if len(picked) == 25:
                break
        if len(picked) != 25:
            raise SystemExit(f"{task}: only {len(picked)} eligible questions")
        chosen.extend(picked)

    assert len(chosen) == len(selected_images) == 100
    metadata_path = out / "metadata.jsonl"
    write_jsonl(metadata_path, chosen)
    manifest = {
        "purpose": "Fresh, small pilot test; excluded from the v3 test.",
        "selection_seed": SELECTION_SEED,
        "source": "GQA val_balanced_questions.json",
        "counts": dict(Counter(row["task"] for row in chosen)),
        "unique_images": len(selected_images),
        "excluded_metadata": {str(path.relative_to(ROOT)): sha256_file(path) for path in previous},
        "metadata_sha256": sha256_file(metadata_path),
    }
    if args.extract_images:
        report = extract_images(cfg, selected_images)
        if report["missing"] or report["corrupt"]:
            raise SystemExit(f"Image extraction failed: {report}")
        manifest["image_report"] = report
    write_json(out / "manifest.json", manifest)
    print(f"Selected {len(chosen)} questions on {len(selected_images)} images: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
