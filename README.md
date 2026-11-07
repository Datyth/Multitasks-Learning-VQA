# Multi-Task VQA Research Pilot

**Research question:** to what extent do VQA question-type labels predict positive and negative transfer during
multi-task adaptation of vision-language models, compared with shared visual reasoning operations?

This is a small controlled pilot: Qwen3.5-4B, QLoRA through `ms-swift`, GQA, one GPU, about 24 hours. It proposes no new
architecture. It is motivated by VQACL, task-transfer and task-grouping work, but it does not reproduce any of those papers.

Generated datasets, checkpoints, and reports are local artifacts excluded from Git. Reproduce them with the commands below.

## Design

Tasks are defined by the GQA structural type and by whether the functional program contains a `relate` step:

| Task | Structural type | `relate` step | Answer format |
|---|---|---|---|
| T1 | query | no | open |
| T2 | query | yes | open |
| T3 | verify | no | yes/no |
| T4 | verify | yes | yes/no |

| Experiment | Adapters | Trained on | Evaluated on | Priority |
|---|---|---|---|---|
| E0 pretrained | none | nothing | T1-T4 | mandatory |
| E1 single-task | `e1_t1` ... `e1_t4` | one task each | T1-T4 each (4×4 matrix) | mandatory |
| E2 naive joint | `e2_joint` | T1-T4 | T1-T4 | mandatory |
| E3 question type | `e3_query`, `e3_verify` | T1+T2, T3+T4 | routed by oracle metadata | exploratory |
| E4 operation | `e4_no_relate`, `e4_relate` | T1+T3, T2+T4 | routed by oracle metadata | exploratory |

**Execution priority:** E0 → E1 → E2 first. Run E3 → E4 only if time remains.

**Measures**

- Directed transfer: `T[i→j] = Acc_j(M_i) − Acc_j(M_0)`.
- Joint interference: `I_i = Acc_i(M_joint) − Acc_i(M_i)`.
- Grouping comparison: E4 − E3. Differences carry paired bootstrap 95% CIs over test questions.

**Evaluation protocol:** the train pool and internal validation come from the official balanced **train** split (split by
image). The held-out pilot test comes from the official balanced **val** split. GQA's test labels are hidden and unused, so
**results are not official GQA test benchmark results.** Internal validation only reports a loss at the end of training; it
is not used to select anything.

**Interpretation limits**

- `query` vs `verify` is confounded with answer format. E3's adapters each see one format, while E4's must learn both, so
  E3 vs E4 also tests format specialization. Per-format macros and yes/no prediction rates are reported for this reason.
- E3/E4 use two adapters and E2 uses one, so a gain over E2 may reflect capacity or specialization rather than grouping.
  E3 vs E4 is the closer comparison.
- E3/E4 use oracle routing from dataset metadata. They are not automatic routing systems.
- `relate` is a coarse diagnostic. Four groups show feasibility and cannot establish which taxonomy generally predicts
  transfer. A positive E4 result does not validate the general reasoning-operation hypothesis.
- Directed transfer does not necessarily predict joint interference. One seed gives no estimate of training variation.

## Repository

```text
config.yaml           all settings: paths, targets, task rules, training budget
requirements.txt      version constraints and their sources
src/common.py         experiment definitions, task rules, normalization, metrics
src/prepare.py        audit, build, extract-images, train-args, run-meta, check-swift
src/evaluate.py       sequential inference and scoring for one experiment at a time
src/report.py         reports/ and the --estimate budget tool
scripts/train.sh      one experiment -> one `swift sft` call; never chains
tests/test_pipeline.py  synthetic checks; loads no weights or real data
```

## Manual commands, in order

None of these runs automatically. Run them from the repository root on the GPU machine.

**On a shared machine,** Steps 6-12 can instead go through `scripts/run_queue.sh`, which you start by hand. It runs the same
commands one at a time and waits for a free GPU before each GPU step. See `bash scripts/run_queue.sh --help`.

```bash
tmux new -s vqa
conda activate datpt_ml
bash scripts/run_queue.sh smoke        # then review the Step 6 outputs
bash scripts/run_queue.sh mandatory exploratory
```

### Step 1: Environment setup

The project uses the existing `datpt_ml` env (Python 3.11). To create it elsewhere, run
`conda create -n datpt_ml python=3.11 -y` first.

```bash
conda activate datpt_ml
which python    # must end in envs/datpt_ml/bin/python

# PyTorch first. torch 2.9.1+cu128 is known to see the GPUs on this machine (driver 595.99.02, CUDA 13.2).
# On another machine, check `nvidia-smi` and pick the matching index from https://pytorch.org/get-started/locally/
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Verify versions, GPU visibility, and every generated flag against the installed ms-swift
python -c "import sys, torch, torchvision, transformers, peft, bitsandbytes; print(sys.version); print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print('transformers', transformers.__version__, 'peft', peft.__version__, 'bnb', bitsandbytes.__version__)"
swift --version 2>/dev/null || pip show ms-swift | head -2
python src/prepare.py check-swift
python -m pytest tests -q
```

**Optional FLA:** the ms-swift guide suggests `flash-linear-attention` for faster Qwen3.5 kernels and recommends Python 3.12
for it. ms-swift falls back to torch kernels for single-sequence batches (micro-batch 1 here), so it is not required. If you
install it, do so before all runs and keep the environment fixed across experiments.

### Step 2: Download GQA

Requires about 22 GB. Neither archive is fully extracted.

```bash
mkdir -p data/raw
wget -c -P data/raw https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
wget -c -P data/raw https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
```

### Step 3: Audit GQA

```bash
python src/prepare.py audit
```

This streams annotations from the ZIP and writes `data/processed/audit.{json,md}`: counts, images, types, operation
frequencies, answer and yes/no distribution, program length, missing annotations and overlaps. **Read it before building.**
Empty or small groups are reported and never redefined automatically; revise `config.yaml` if needed.

### Step 4: Prepare the dataset

```bash
python src/prepare.py build
```

This selects samples deterministically (seed 42), extracts only the referenced images, validates each one, and writes the
outputs below. To stop before image extraction, use `build --skip-extract`. To retry extraction alone, use
`python src/prepare.py extract-images`, then re-run `build`.

```text
data/processed/metadata.jsonl       question id, image id, question, answer, task, types, operations, split, image path
data/processed/manifest.json        counts, hashes, per-experiment step counts, config snapshot
data/processed/train/<id>.jsonl     ms-swift multimodal training data
data/processed/val/<id>.jsonl       internal validation data
data/processed/images/              only the referenced images
```

Exit code 2 means too few samples: nothing is written, and the message gives actual counts and the largest balanced
`data.targets`. Exit code 3 means missing or corrupt images: see `data/processed/image_report.json`. Training JSONL is not
written in that case, and no sample is replaced.

### Step 5: Download the model

```bash
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

The training and evaluation scripts refuse to run if this path is missing, so nothing downloads implicitly.

### Step 6: Smoke test and profile

Loads weights and uses GPU 0. Run each by hand.

```bash
bash scripts/train.sh e1_t1 --dry-run                   # prints the resolved command; loads nothing
bash scripts/train.sh e1_t1 --smoke                     # 3 optimizer steps -> runs/smoke/e1_t1
python src/evaluate.py --experiment e0 --limit 5        # 5 per task, pretrained -> runs/smoke/e0
python src/evaluate.py --experiment e1_t1 --limit 5     # 5 per task, smoke adapter -> runs/smoke/e1_t1
```

Check these before continuing:

- `runs/smoke/e0/predictions.jsonl`: raw outputs are short answers with no thinking text.
- `runs/smoke/e1_t1/logging.jsonl`: `memory(GiB)` (peak) and `train_speed(s/it)` (time per optimizer step).
- `runs/smoke/e0/metrics.json`: `runtime.seconds_per_example`, `load_seconds` and `peak_gpu_memory_mib`.

Then estimate whether the budget fits, using those measured timings or your own (`--sec-per-step`, `--sec-per-example`,
`--load-seconds`):

```bash
python src/report.py --estimate --hours-available 24
```

If E0-E2 do not fit, lower `train.examples_per_task` and/or `evaluation.samples_per_task` for **all** experiments, then
re-run Step 4. Smoke reruns need `--force` (train) or `--overwrite` (evaluate). Smoke output never enters the report.

**If QLoRA fails** on compatibility or VRAM, keep the actual error. Do not silently switch models, quantization or task
definitions. A documented change (for example `Qwen/Qwen3.5-2B` in `config.yaml`) must apply to every experiment and be
stated in the report.

### Step 7: Train E1

```bash
bash scripts/train.sh e1_t1
bash scripts/train.sh e1_t2
bash scripts/train.sh e1_t3
bash scripts/train.sh e1_t4
```

### Step 8: Train E2

```bash
bash scripts/train.sh e2_joint
```

### Step 9: Train E3 (exploratory)

```bash
bash scripts/train.sh e3_query
bash scripts/train.sh e3_verify
```

### Step 10: Train E4 (exploratory)

```bash
bash scripts/train.sh e4_no_relate
bash scripts/train.sh e4_relate
```

Each job starts from the same pretrained checkpoint. `train.sh` refuses to overwrite an existing run without `--force`.

### Step 11: Evaluation

Single experiments:

```bash
python src/evaluate.py --experiment e0
python src/evaluate.py --experiment e1_t1
python src/evaluate.py --experiment e1_t2
python src/evaluate.py --experiment e1_t3
python src/evaluate.py --experiment e1_t4
python src/evaluate.py --experiment e2_joint
python src/evaluate.py --experiment e3_query
python src/evaluate.py --experiment e3_verify
python src/evaluate.py --experiment e4_no_relate
python src/evaluate.py --experiment e4_relate
```

Or by group, where `e1`, `e2`, `e3` and `e4` are shorthand. To evaluate E0 plus every finished adapter:

```bash
python src/evaluate.py --experiment all --dry-run   # shows the plan; loads nothing
python src/evaluate.py --experiment all
```

Adapters that are missing or stopped early are skipped and reported. Finished evaluations are skipped unless `--overwrite`
is passed. An interrupted evaluation resumes from `predictions.jsonl`. Evaluate E0 early, because everything else is
compared against it.

### Step 12: Generate the report

```bash
python src/report.py
```

This works with any subset of experiments. Anything not run shows as `not run`, and the research decision stays `Pending`
until E3 and E4 are both complete.

## Version v2_1000train_500test (larger sample, three training seeds)

`config.v2_1000train_500test.yaml` keeps v1's model, QLoRA settings, prompt, T1-T4 definitions, metrics and E0-E4
logic. Only these settings change: 1,000 train, 200 internal-validation and 500 test questions per task, training
seeds 42/43/44, GPU 1, and separate output paths. v1 files (`data/processed`, `runs/<id>`, `reports/`) are never written.

| | v1 (`config.yaml`) | v2 (`config.v2_1000train_500test.yaml`) |
|---|---|---|
| Data | `data/processed/` | `data/v2_1000train_500test/` |
| Adapters, predictions | `runs/<id>/` | `runs/v2_1000train_500test/seed<N>/<id>/`; E0 in `.../e0/` |
| v1 adapters on the v2 test | | `runs/v2_1000train_500test/v1_400train/<id>/` |
| Report | `reports/` | `reports/v2_1000train_500test/` |

**Test set.** It is drawn from GQA balanced val after removing every question and image that v1 used, in any split.
The build fails if any of them reappears. Train/validation/test are image-disjoint. The test is only scored, so no
hyperparameter, checkpoint or early stop depends on it: training uses fixed step counts, and validation loss comes from
the internal-validation split.

**Equal exposure.** Each task contributes the same 1,000 examples once in every experiment: E3/E4 adapters take
250 steps, E2 takes 500 and E1 takes 125. All other hyperparameters are identical for every experiment and seed.
Training seeds change only LoRA initialisation and data order; the dataset seed stays 42.

**CIs.** E4 − E3 is a paired difference on identical questions. The 95% CI comes from a bootstrap over test images, so
questions on the same image (also across tasks) are resampled together. The seed-pooled contrast averages each
question's correctness over seeds first. The report also lists the per-seed differences and their spread.

All commands run from the repository root in `conda activate datpt_ml`. Every GPU step uses GPU 1 (`gpu: 1` in the
config). None of the commands below runs automatically.

```bash
V2=config.v2_1000train_500test.yaml

# 1. Data (CPU; streams the local GQA ZIPs, extracts about 6,800 images; nothing is downloaded)
python src/prepare.py --config $V2 audit          # availability per task -> data/v2_1000train_500test/audit.md
python src/prepare.py --config $V2 build          # prints availability after excluding v1, then samples
python -m pytest tests -q

# 2. Lightweight checks (no model loaded)
bash scripts/train.sh e3_query --config $V2 --seed 42 --dry-run
python src/evaluate.py --config $V2 --experiment all --dry-run
python src/report.py --config $V2 --estimate --hours-available 24

# 3. Pretrained baseline and existing v1 E3/E4 adapters on the new test (no retraining)
python src/evaluate.py --config $V2 --experiment e0
python src/evaluate.py --config $V2 --experiment e3 --adapters-from v1_400train
python src/evaluate.py --config $V2 --experiment e4 --adapters-from v1_400train

# 4. Train and evaluate E1-E4 for one seed (repeat with SEED=43 and SEED=44)
SEED=42
for e in e3_query e3_verify e4_no_relate e4_relate e2_joint e1_t1 e1_t2 e1_t3 e1_t4; do
  bash scripts/train.sh $e --config $V2 --seed $SEED || break
done
for e in e3 e4 e2_joint e1; do
  python src/evaluate.py --config $V2 --experiment $e --seed $SEED
done

# 5. Report (works with any subset; the decision waits for all three seeds)
python src/report.py --config $V2
```

The same GPU steps, queued on GPU 1 (`v2` = step 3 and E1-E4 for seed 42; `v2-seeds` = seeds 43 and 44):

```bash
bash scripts/run_queue.sh v2 v2-seeds --gpus 1 --dry-run
bash scripts/run_queue.sh v2 v2-seeds --gpus 1
```

The v2 report adds directed transfer (E1 vs E0, mean over seeds) and MTL interference (E2 − E1, seed-pooled CI)
once E1 and E2 exist. `train.sh` and `evaluate.py` refuse to overwrite finished runs unless `--force` or `--overwrite` is passed.

## Version v3_crossed_open (one-seed crossed pilot)

`config.v3_crossed_open.yaml` defines four open-answer `query` tasks from GQA detailed types `directOf`/`directWhich`
crossed with absence/presence of a `relate` program step. This keeps answer format fixed while comparing grouping by
detailed question type (E3) with grouping by operation (E4). The legacy adapter IDs `e3_query` and `e3_verify` refer to
the `directOf` and `directWhich` groups in this version. E2 is the single-adapter joint baseline.

The v3 test has 500 questions per task from GQA balanced val, excluding every question and image used in v1, v2, or the
25-per-task quick holdout. All output paths are separate. A single training seed is a pilot, not a stability estimate.

```bash
conda activate datpt_ml
V3=config.v3_crossed_open.yaml
python scripts/prepare_quick_holdout.py  # recreates the 100 selected question IDs; images are optional
python src/prepare.py --config "$V3" build
bash scripts/run_queue.sh v3-one-seed --gpus 0 --stable-polls 1 --poll-seconds 10
```

The queue runs E3/E4 first and writes an interim report, then runs E2 and updates `reports/v3_crossed_open/report.md`.
It skips completed steps on restart. The dataset build needs the local GQA archives and the previous versions' metadata;
the GPU steps need the local Qwen checkpoint. Earlier results remain untouched.

## Outputs

| Path | Content |
|---|---|
| `runs/<id>/checkpoint-<steps>/` | LoRA adapter |
| `runs/<id>/{config.yaml,run_config.json,train.log,logging.jsonl}` | config snapshot, resolved command, environment, status, logs |
| `runs/<id>/predictions.jsonl` | raw and normalized predictions for every question |
| `runs/<id>/metrics.json` | per-task, macro and pooled accuracy, invalid rate, runtime, config snapshot |
| `reports/results.csv`, `transfer_matrix.csv`, `transfer_heatmap.png`, `report.md` | required report files |
| `reports/summary.json` | machine-readable transfer, interference and decision values |

## Scoring and decision rule

**Scoring:** exact match after normalization. Normalization lowercases, removes punctuation and think blocks, treats hyphens
as separators, drops articles and preserves numbers. Number words are not mapped to digits (`two` ≠ `2`), and there is no
semantic matching. Empty output counts as invalid and incorrect. Macro average weights T1-T4 equally.

**Research decision (fixed before running):** based on the E4 − E3 macro difference and its 95% CI.

- **Promising:** the CI is above zero.
- **Not supported:** the CI is below zero.
- **Inconclusive:** the CI includes zero, or the invalid generation rate exceeds 10%.

The rule's output is mechanical and must be reviewed against the report's limitations. The CI covers test-question sampling
only, not seed-to-seed variation.

## Deviations and verified details

- **Repository layout:** files live at the repository root instead of `vqa-mtl/`. `src/common.py` was added because all
  three scripts need the same definitions.
- **Python 3.11** instead of the preferred 3.12, as requested for the `datpt_ml` env. The guide's 3.12 recommendation
  concerns the optional FLA package. `ms-swift` 4.6.0 and the other pinned packages declare 3.11 support.
- **`lora_alpha` 16** follows this project's specification; the guide's example uses 32.
- **Step counts:** `max_steps = n_train // effective_batch_size`, giving 50/100/200 at defaults. A partial last batch is
  dropped if the budget is not divisible.
- **Flags** were checked against the ms-swift 4.6.0 wheel source, not a working install. Run `check-swift` in Step 1 to
  confirm against the installed version.
- **GPU selection:** `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set so that `gpu: 0` matches `nvidia-smi` index 0.
