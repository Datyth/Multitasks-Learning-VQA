# Implement a Minimal Multi-Task VQA Research Pilot

You are a senior Machine Learning Research Engineer experienced with PyTorch, vision-language models, Hugging Face, and reproducible experiments.

Implement a **small, scientifically sound VQA research project from scratch**, using the specifications below.

Your primary responsibility is to **write the code, documentation, and executable commands**. Do not start expensive operations.

## 1. Project Context

### Research motivation

We are investigating whether VQA question-type labels represent meaningful learning-task relationships, compared with shared visual reasoning operations.

Our primary research question is:

**"To what extent do VQA question-type labels predict positive and negative transfer during multi-task adaptation of vision-language models, compared with shared visual reasoning operations?"**

Previous work such as VQACL organizes VQA tasks around question categories. GQA provides both question-type labels and functional-program annotations.

Our goal is to first establish reliable VQA adaptation baselines, identify possible multi-task interference, and then explore whether a simple alternative task grouping offers benefits.

We are **not proposing a new model architecture at this stage**.

### Research constraints

- Hardware: **ONE NVIDIA RTX GPU with 24GB VRAM**.
- Total experimental GPU time budget: approximately **24 hours**.
- Main model: `Qwen/Qwen3.5-4B`.
- Fallback only if necessary: `Qwen/Qwen3.5-2B`.
- Main dataset: GQA.
- Training: LoRA or QLoRA.
- Main framework: `ms-swift`.
- One deterministic dataset construction, shared across all experiments.
- No distributed training.
- No cloud inference APIs.

Prioritize **correctness, simplicity, reproducibility, and research validity** over engineering complexity.

## 2. CRITICAL EXECUTION RULES

You are authorized to inspect the repository, create files, modify code, and execute lightweight checks.

However, you **MUST NOT automatically execute**:

- Dataset downloads.
- Model checkpoint downloads.
- Training or fine-tuning jobs.
- Full-dataset evaluation.
- Large archive extraction.
- Any other GPU-intensive or long-running operation.

Do not run shell scripts merely to verify that they exist.

Instead, implement the required scripts and provide exact commands for the user to execute manually.

You MAY run lightweight operations such as:

- Inspecting repository files.
- Checking Python and CUDA metadata.
- Checking available GPU VRAM with `nvidia-smi`.
- Python syntax checks.
- Unit tests using tiny synthetic fixtures.
- CLI `--help` verification.
- Dependency and version inspection, if packages already exist.

Do not use real GQA samples or model weights in tests unless already available locally.

Do not silently install packages or download resources.

Do not create background jobs, scheduled tasks, or automatic pipelines.

**At the end, clearly list every manual command needed, in the correct execution order.**

## 3. References and Source Verification

Read the following sources before implementing model-specific or dataset-specific logic.

### Model

Qwen3.5-4B official checkpoint:

https://huggingface.co/Qwen/Qwen3.5-4B

Qwen3.5-2B fallback:

https://huggingface.co/Qwen/Qwen3.5-2B

Hugging Face Transformers:

https://github.com/huggingface/transformers

### Fine-tuning framework

ms-swift:

https://github.com/modelscope/ms-swift

Qwen3.5/Qwen3.6/Qwen3.8 training guide:

https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_8-Best-Practice.md

Multimodal dataset format:

https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html

CLI parameters:

https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html

QLoRA training example:

https://github.com/modelscope/ms-swift/blob/main/examples/train/qlora/bnb/train.sh

### Dataset

GQA official website:

https://cs.stanford.edu/people/dorarad/gqa/

Official download and annotation specification:

https://cs.stanford.edu/people/dorarad/gqa/download.html

Questions archive:

https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip

Images archive:

https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip

GQA paper:

https://arxiv.org/abs/1902.09506

### Related research

VQACL, CVPR 2023:

https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_VQACL_A_Novel_Visual_Question_Answering_Continual_Learning_Setting_CVPR_2023_paper.html

Understanding Task Transfer in Vision-Language Models, CVPR 2026:

https://openaccess.thecvf.com/content/CVPR2026/html/Sachdeva_Understanding_Task_Transfer_in_Vision-Language_Models_CVPR_2026_paper.html

Visual Question Answering as a Multi-Task Problem:

https://arxiv.org/abs/2007.01780

Which Tasks Should Be Learned Together in Multi-task Learning?

https://proceedings.mlr.press/v119/standley20a.html

Use these references to understand the experimental motivation.

Do not claim exact paper reproduction unless the experiment actually follows that paper's model, data, and protocol.

Our proposed experiments are controlled pilot baselines motivated by prior work, not exact numerical reproductions of VQACL or other papers.

If documentation has changed, use the current verified API and explain any necessary deviation.

## 4. Keep the Repository Minimal

Use the following structure as a guide:

```text
vqa-mtl/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── prepare.py
│   ├── evaluate.py
│   └── report.py
├── scripts/
│   └── train.sh
├── tests/
│   └── test_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── runs/
└── reports/
```

Do not create empty Python packages unnecessarily.

Use `.gitignore` to exclude datasets, model checkpoints, caches, logs, and generated results.

Prefer ordinary Python scripts and argparse.

Avoid:

- Hydra.
- Lightning.
- MLflow.
- Weights & Biases.
- Docker or Kubernetes.
- Distributed training infrastructure.
- Custom Trainer implementations.
- Complex configuration inheritance.
- Generic plugin systems.
- Unnecessary abstraction layers.

Use `ms-swift` for training instead of reimplementing its training loop.

Create extra files only when there is a concrete need.

## 5. Environment

Prefer Python 3.12.

Use PyTorch with the appropriate CUDA build for the installed driver and GPU.

Use compatible versions of:

- PyTorch and torchvision.
- Transformers.
- ms-swift.
- PEFT.
- bitsandbytes.
- accelerate.
- qwen-vl-utils, if required.
- Pillow.
- NumPy.
- pandas.
- matplotlib.
- ijson, if required for memory-efficient JSON parsing.
- pytest.

Write `requirements.txt` with compatible, documented version constraints.

Document how to create the Conda environment.

Do not hardcode a CUDA-specific PyTorch installation command without checking compatibility with the actual environment.

Do not make FlashAttention a mandatory dependency unless the selected model/training stack requires it.

The environment must support Qwen3.5 multimodal inference and QLoRA training.

## 6. Dataset Preparation

### 6.1 Downloading

Provide explicit commands in README for downloading:

- `questions1.2.zip`
- `images.zip`

These files are large. The images archive is approximately 20GB.

Do not download or extract them automatically during implementation.

The user should manually execute the download commands.

### 6.2 Read GQA annotations

Inspect the real GQA schema.

Relevant fields include:

```text
questionId
imageId
question
answer
isBalanced
types.structural
types.semantic
types.detailed
semantic[].operation
```

Question IDs are dictionary keys in the official annotation files.

Use the official balanced question splits:

```text
train_balanced_questions.json
val_balanced_questions.json
```

Do not assume that GQA's hidden test labels are available.

Use the official training split for training and internal validation, and the official validation split for held-out pilot testing.

Clearly describe this evaluation protocol in README and the final report. Do not present it as a result on the official GQA test benchmark.

Support reading the JSON files directly from the downloaded ZIP archive, if practical.

Avoid extracting all question files unnecessarily.

### 6.3 Initial task definitions

Build four candidate task groups:

| Group | Structural type | Contains `relate` operation |
|---|---|---|
| T1 | query | No |
| T2 | query | Yes |
| T3 | verify | No |
| T4 | verify | Yes |

Determine operation membership using the actual `semantic` functional-program steps, not only `types.semantic`.

For example, a question belongs to the `relate` group when at least one step has:

```python
step["operation"] == "relate"
```

This is a **coarse operation-based diagnostic**, not a complete measure of reasoning similarity.

Before sampling, implement a dataset audit to check:

- Sample count for each task.
- Number of unique images.
- Structural and semantic question types.
- Functional-program operation frequencies.
- Answer distribution, including yes/no proportions.
- Average program length.
- Missing annotations.
- Whether group definitions overlap or produce empty groups.

Important: `query` versus `verify` can strongly correlate with answer format. Record this explicitly as a potential confound.

Do not silently change group definitions when a group is empty or too small.

Report the issue and allow the user to revise the configuration.

### 6.4 Sampling

Use the following targets:

| Split | Samples per task |
|---|---:|
| Train pool | 1,000 |
| Internal validation | 100 |
| Held-out pilot test | 150 |

These are targets, not assumed dataset counts.

If the available samples are insufficient, report the actual counts and provide a documented smaller balanced option.

Use a fixed random seed, initially 42.

Prevent image leakage:

- No shared `imageId` between train and internal validation.
- No shared `imageId` between training-related splits and the held-out pilot test.
- Validate that there are no duplicate question IDs across splits.

Do not perform naive random question-level splitting when multiple questions refer to the same image.

### 6.5 Image extraction

Do not extract the entire images archive if only a small subset of images is required.

After selecting question IDs:

1. Collect the required image IDs.
2. Inspect the images archive.
3. Extract only referenced images.
4. Validate every extracted image file.
5. Report missing or corrupt files.

Support manually running the image extraction as an explicit user command.

Do not trigger the extraction implicitly during unrelated operations.

### 6.6 Training format

Create ms-swift-compatible multimodal JSONL.

Example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>Answer the question using a short answer only: What color is the car?"
    },
    {
      "role": "assistant",
      "content": "red"
    }
  ],
  "images": [
    "/absolute/path/to/image.jpg"
  ]
}
```

The same user prompt must be used for every experiment.

Do not reveal task group labels, reasoning operations, or ground-truth answers to the model at inference time.

Store question metadata separately for evaluation and analysis.

Preserve:

- Question ID.
- Image ID.
- Original question.
- Ground-truth answer.
- Task group.
- Structural and semantic types.
- Functional-program operations.
- Data split.

## 7. Fixed Training Budget

We must be able to finish a useful pilot within one day on one GPU.

Use a small, fixed training budget initially.

Default:

```yaml
model: Qwen/Qwen3.5-4B
seed: 42
gpu: 0

train:
  examples_per_task: 400
  effective_batch_size: 8
  learning_rate: 0.0001
  lora_rank: 8
  lora_alpha: 16
  max_length: 1024
  image_max_token_num: 256
  quant_bits: 4

evaluation:
  max_new_tokens: 16
  temperature: 0
  samples_per_task: 150
```

The configuration should be easy to change.

Use deterministic sampling so every task contributes approximately 400 training examples.

With effective batch size 8:

- One-task training: approximately 50 optimizer steps.
- Two-task training: approximately 100 optimizer steps.
- Four-task training: approximately 200 optimizer steps.

The number of examples per task, model checkpoint, output format, and adapter configuration must be consistent across experiments.

The pilot may need smaller budgets after the user runs a profiling command. Do not assume these jobs will fit within 24 hours without measurement.

Implement a configurable smoke-test mode with at most 2–5 optimizer steps.

The agent must not execute that smoke test automatically because it loads model weights and runs the GPU.

## 8. Training Implementation

Use the official ms-swift `swift sft` command.

Start with:

- Qwen3.5-4B.
- QLoRA using bitsandbytes NF4.
- Single GPU.
- Micro-batch size 1.
- Gradient accumulation 8.
- LoRA rank 8.
- Frozen vision encoder.
- Frozen visual aligner if supported.
- Short answer supervised fine-tuning.
- Non-thinking inference/training configuration according to the Qwen3.5 guide.
- Minimal checkpoints.
- No external experiment tracking.

Verify the exact CLI arguments against the installed ms-swift version.

In particular, check:

```text
--model
--dataset
--val_dataset
--tuner_type
--quant_method
--quant_bits
--freeze_vit
--freeze_aligner
--lora_rank
--lora_alpha
--target_modules
--max_steps
--gradient_accumulation_steps
--per_device_train_batch_size
--output_dir
```

Use the official Qwen3.5 guidance for non-thinking output and multimodal image limits.

If QLoRA fails due to compatibility or VRAM issues, report the actual error and document a verified alternative.

Do not silently switch models, quantization methods, or task definitions.

Do not implement custom transformer layers or modify Qwen's architecture.

Create one `scripts/train.sh` accepting an experiment ID, for example:

```bash
bash scripts/train.sh e1_t1
```

The script should translate the experiment ID into the correct dataset and output directory, then call ms-swift.

Do not automatically chain all experiments.

The user must be able to run each job separately.

Save the configuration and training logs alongside each adapter.

## 9. Required Experiments

Implement the following five experiments.

### E0: Pretrained Baseline

Evaluate the original Qwen3.5-4B without adaptation.

Evaluate separately on T1, T2, T3, and T4.

Output:

```text
runs/e0/predictions.jsonl
runs/e0/metrics.json
```

Purpose:

Establish the initial performance on all tasks.

### E1: Single-Task Adaptation and Directed Transfer

Train four independent adapters:

```text
e1_t1
e1_t2
e1_t3
e1_t4
```

Each adapter is trained only on its assigned task.

All four adapters must start from the same pretrained checkpoint.

Evaluate **each adapter on all four tasks**, not only its training task.

This creates a 4 × 4 performance matrix.

Directed transfer is defined as:

\[
T_{i\rightarrow j}
=
Acc_j(M_i)-Acc_j(M_0)
\]

Here:

- \(M_0\): original pretrained model.
- \(M_i\): model adapted on source task \(i\).
- \(Acc_j\): accuracy on target task \(j\).

Report the full matrix, including off-diagonal values.

Do not call source-only cross-task evaluation "joint MTL interference".

### E2: Naive Multi-Task Learning

Train one LoRA adapter on the balanced union of all four tasks.

Experiment ID:

```text
e2_joint
```

Use the same model and training settings.

Each task contributes the same number of training examples as in E1.

Evaluate on T1–T4.

Compare joint training with the corresponding single-task adapter:

\[
I_i =
Acc_i(M_{\mathrm{joint}})
-
Acc_i(M_i)
\]

Interpretation:

- Positive: joint training helps task \(i\) relative to its single-task baseline.
- Negative: joint training hurts task \(i\) relative to its single-task baseline.

This is our first empirical multi-task interference analysis.

### E3: Question-Type Grouping

Create two independent adapters:

```text
e3_query  = T1 + T2
e3_verify = T3 + T4
```

Both adapters start from the same pretrained checkpoint.

Each adapter sees 400 examples from each assigned task.

At evaluation:

- Route T1 and T2 to the Query adapter.
- Route T3 and T4 to the Verify adapter.

This is an oracle-routing diagnostic baseline, because the correct task group is known from dataset metadata.

Do not describe it as automatic routing.

### E4: Operation-Based Grouping

Create two independent adapters:

```text
e4_no_relate = T1 + T3
e4_relate    = T2 + T4
```

Use the same training budgets and configuration as E3.

At evaluation:

- Route T1 and T3 to the non-relate adapter.
- Route T2 and T4 to the relate adapter.

Again, this uses oracle metadata for evaluation routing.

This tests whether a very simple operation-aware partition can improve adaptation results.

It does not establish that `relate` is the optimal grouping criterion.

### Experiment priority

Implement all experiment definitions.

But document the execution order as:

**Mandatory:** E0 → E1 → E2

**Exploratory:** E3 → E4

If the one-day budget is insufficient, E3/E4 may remain unexecuted.

Never fabricate missing results.

## 10. Evaluation

Implement `src/evaluate.py`.

Use either the verified ms-swift inference API or the corresponding supported Hugging Face/PEFT inference API.

Avoid reimplementing model internals.

Requirements:

- Load one model/adapter at a time.
- Run inference sequentially on one GPU.
- Disable thinking where supported.
- Use deterministic decoding.
- Use the same prompt for all experiments.
- Generate short answers.
- Store raw outputs before normalization.
- Normalize predictions consistently.
- Use exact-match accuracy for this pilot.
- Report invalid or empty generations.
- Do not use ground-truth information during generation.

Example prediction record:

```json
{
  "question_id": "123",
  "task": "T1",
  "question": "What color is the car?",
  "ground_truth": "red",
  "prediction_raw": "red",
  "prediction_normalized": "red",
  "correct": true
}
```

For normalization:

- Lowercase.
- Strip whitespace.
- Normalize punctuation.
- Handle simple articles consistently.
- Preserve meaningful numeric answers.
- Do not use semantic similarity to automatically declare an answer correct.

Report:

1. Accuracy for each task.
2. Macro-average accuracy across T1–T4.
3. Pooled overall accuracy.
4. Directed transfer matrix.
5. Joint-training interference values.
6. E3 versus E4 performance difference.
7. Invalid generation rate.
8. Basic training/runtime information.

Make sure macro-average gives equal weight to the four tasks.

Preserve the distinction between open-answer Query tasks and binary Verify tasks.

Avoid relying only on pooled accuracy, because answer-format differences can distort the interpretation.

## 11. Reporting

Implement `src/report.py`.

Read saved experiment outputs and generate:

```text
reports/
├── results.csv
├── transfer_matrix.csv
├── transfer_heatmap.png
└── report.md
```

The report should contain:

### A. Experimental Setup

- Model and checkpoint.
- Dataset.
- Task definitions.
- Data counts.
- Training configuration.
- GPU information.
- Number of training steps.
- Random seed.
- Which experiments actually completed.

### B. Main Results

One table:

| Experiment | T1 | T2 | T3 | T4 | Macro Avg |
|---|---|---|---|---|---|
| E0 | | | | | |
| E1 single-task | | | | | |
| E2 joint | | | | | |
| E3 question-type | | | | | |
| E4 operation grouping | | | | | |

For E1, use the target-specific single-task result on the diagonal.

### C. Directed Transfer

Show the 4 × 4 matrix.

Explain positive, negative, and asymmetric relationships.

### D. MTL Interference

Compare E2 with E1 for each target task.

### E. Preliminary Idea Assessment

Compare E3 against E4.

Report both macro-average and per-task effects.

Do not claim architectural superiority from one small run.

### F. Limitations

Explicitly discuss:

- Question type versus answer-format confounding.
- Small number of tasks.
- Limited training steps.
- Single checkpoint.
- Single random seed.
- Coarse operation grouping based only on `relate`.
- Oracle routing for E3/E4.
- Difference between source-only transfer and joint MTL.
- No official benchmark reproduction.

### G. Research Decision

Give one of three evidence-based conclusions:

- **Promising:** Continue with larger experiments.
- **Inconclusive:** Improve dataset/task design or repeat experiments.
- **Not supported:** Reconsider the proposed research direction.

If experiments have not been run, produce a report template without invented scores.

## 12. Reproducibility and Sanity Checks

Implement minimal tests using synthetic annotations and tiny dummy images.

Check that:

1. Task classification follows the intended structural label and operation rules.
2. Every selected sample belongs to exactly one group.
3. No training/test image overlap exists.
4. No question ID is duplicated across splits.
5. All selected image paths exist.
6. All generated training records have valid multimodal JSONL format.
7. E1–E4 use the same per-task example budgets.
8. Answer normalization behaves consistently.
9. Metrics and transfer-matrix calculations are correct.
10. Reports handle incomplete experiments gracefully.

Do not load Qwen checkpoints for these tests.

Use fixed random seeds.

Write configuration snapshots for every run.

## 13. Required CLI Commands and Documentation

README must provide commands in the exact order below.

### Step 1: Environment setup

Create and activate the Conda environment.

Install dependencies.

Verify Python, PyTorch, CUDA, and the installed framework versions.

### Step 2: Download GQA

Provide explicit commands to download the question and image archives.

Do not automatically run them.

### Step 3: Audit GQA

Example interface:

```bash
python src/prepare.py audit
```

This should inspect task distributions and output a summary.

### Step 4: Prepare the dataset

Example interface:

```bash
python src/prepare.py build
```

This should create reproducible splits, extract only the required images, and write training/evaluation metadata.

### Step 5: Download the model

Document the Hugging Face CLI command to download Qwen3.5-4B into `models/`.

Do not download it automatically.

### Step 6: Smoke test

Provide a command to train only 2–5 steps, and a command to evaluate only a few examples.

Both must require manual execution.

### Step 7: Train E1

Provide individual commands:

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

### Step 9: Train E3

```bash
bash scripts/train.sh e3_query
bash scripts/train.sh e3_verify
```

### Step 10: Train E4

```bash
bash scripts/train.sh e4_no_relate
bash scripts/train.sh e4_relate
```

### Step 11: Evaluation

Provide commands to evaluate the pretrained model, individual adapters, and all completed experiments.

Evaluation must support selecting an experiment ID.

Example:

```bash
python src/evaluate.py --experiment e0
python src/evaluate.py --experiment e1_t1
```

Do not assume all checkpoints exist.

### Step 12: Generate report

```bash
python src/report.py
```

This must work even when some optional experiments are missing.

Document actual argument names and paths that your implementation supports.

Do not leave pseudocode or commands that do not correspond to the implemented scripts.

## 14. One-Day GPU Budget

The user has only one GPU and approximately one day of experimental time.

Before full training, the user should manually run a small profiling job to determine:

- Peak GPU memory.
- Time per optimizer step.
- Time per evaluation example.
- Expected total training duration.
- Expected total evaluation duration.

Provide a simple way to estimate whether E0–E4 can fit into the remaining time.

Do not invent GPU performance measurements.

Make the experimental budget configurable.

The priority is to complete E0–E2 reliably rather than start every experiment without finishing evaluation.

Avoid frequent checkpoints and unnecessary repeated model downloads.

## 15. Scientific Integrity

Follow these principles throughout implementation and documentation.

- Distinguish an established finding from a research hypothesis.
- Do not call naive joint training a new method.
- Do not call E3/E4 automatic routing systems.
- Do not claim E4 validates the general reasoning-operation hypothesis.
- Do not claim directed transfer necessarily predicts joint interference.
- Do not claim a positive accuracy difference is significant without uncertainty assessment.
- Do not compare different training budgets without documenting the difference.
- Do not silently alter evaluation data or task definitions to improve scores.
- Keep raw predictions for subsequent error analysis.
- Never invent experimental results.

Also document that E3/E4 use two adapters while E2 uses one. Their total adapter capacity differs, so improvements may partly reflect additional specialization/capacity rather than grouping alone.

E3 versus E4 is the closer comparison because both use two adapters with corresponding training budgets.

The four-group design is for feasibility. It is not sufficient on its own to establish which taxonomy generally predicts transfer better.

## 16. Implementation Order

Follow this sequence:

1. Inspect the current repository and available environment.
2. Verify the model, dataset, and framework documentation.
3. Create the minimal project structure.
4. Implement dataset auditing and preparation.
5. Implement experiment dataset generation.
6. Implement the ms-swift training launcher.
7. Implement evaluation.
8. Implement report generation.
9. Add small unit tests.
10. Write the README with all manual commands.
11. Run only lightweight tests and syntax checks.
12. Summarize what has been implemented.

Do not proceed into real data downloads, model downloads, training, or full evaluation.

## 17. Final Deliverables

When implementation is complete, respond with:

**1. Implemented files**

Briefly describe each important file.

**2. Verified functionality**

State which lightweight checks actually passed.

**3. Manual setup commands**

Show environment creation, data download, model download, and dataset preparation commands.

**4. Manual experiment commands**

List the complete training/evaluation commands in the intended order.

**5. Expected outputs**

State where checkpoints, predictions, metrics, and reports will be stored.

**6. Remaining requirements**

Clearly identify anything that cannot be verified without downloading real data or running the GPU.

Keep documentation concise, technical, and accessible to a small academic research team.

**Do not perform the long-running tasks yourself. Stop once the research code, instructions, and lightweight tests are complete.**