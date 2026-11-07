# Multi-task VQA pilot: v3_crossed_open

## Setup

- **Model:** `Qwen/Qwen3.5-4B`, 4-bit NF4 QLoRA; LoRA rank 8, alpha 16, lr 0.0001, effective batch 8, warmup 0.05. Same settings for every experiment and seed.
- **Data:** 1000 train examples per task (one pass), 200 internal-validation and 500 test questions per task. Dataset seed 42; training seeds 42.
- **Equal exposure:** every task contributes the same examples once in every experiment, so E3/E4 adapters take half the optimizer steps of E2.
- **Test:** GQA balanced val split, excluding every question and image of earlier versions. It is not the official GQA test benchmark and was not used to choose any hyperparameter.
- **Routing:** E3/E4 use oracle routing from dataset metadata.

- **Task design:** T1/T2 are `directOf` without/with `relate`; T3/T4 are `directWhich` without/with `relate`. All four have open answers. Legacy adapter IDs `e3_query`/`e3_verify` mean the two detailed-type groups in this run.
- **Inference limit:** with one training seed, CIs cover test-image sampling only; this is a pilot.

| Split | Questions | Images | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|
| train_pool | 4000 | 3877 | 1000 | 1000 | 1000 | 1000 |
| internal_val | 800 | 773 | 200 | 200 | 200 | 200 |
| test | 2000 | 1757 | 500 | 500 | 500 | 500 |

Excluded from the test pool (v1_400train): 5000 questions and 4883 images, including its 600 previously evaluated test questions.

Excluded from the test pool (v2_1000train_500test): 6800 questions and 6515 images, including its 2000 previously evaluated test questions.

Excluded from the test pool (quick_holdout_25per_task): 100 questions and 100 images, including its 100 previously evaluated test questions.

Training runs:

| Seed | Run | Tasks | Status | GPU | Step | Wall s | s/step | Peak GiB |
|---|---|---|---|---|---|---|---|---|
| seed42 | e2_joint | T1+T2+T3+T4 | not run | n/a | n/a | n/a | n/a | n/a |
| seed42 | e3_query | T1+T2 | not run | n/a | n/a | n/a | n/a | n/a |
| seed42 | e3_verify | T3+T4 | not run | n/a | n/a | n/a | n/a | n/a |
| seed42 | e4_no_relate | T1+T3 | not run | n/a | n/a | n/a | n/a | n/a |
| seed42 | e4_relate | T2+T4 | not run | n/a | n/a | n/a | n/a | n/a |

## Accuracy

Exact-match accuracy on the new test. Rows from earlier versions use their existing adapters without retraining.

| Run set | Experiment | T1 | T2 | T3 | T4 | Macro | Invalid |
|---|---|---|---|---|---|---|---|
| none | not run |  |  |  |  |  |  |

## Directed transfer (E1 vs E0)

Not available: needs E0 and all four E1 adapters for at least one seed.

## MTL interference (E2 − E1)

Not available: needs E1 and E2 for at least one seed.

## E4 − E3

Paired difference on identical test questions, with image-clustered bootstrap 95% CIs. The pooled row averages each question's correctness over seeds first; its CI covers test sampling, and the seed spread below shows training variation.

| Run set | T1 | T2 | T3 | T4 | Macro (T1-T4) | directOf macro (T1, T2) | directWhich macro (T3, T4) |
|---|---|---|---|---|---|---|---|
| none | not run | not run | not run | not run | not run | not run | not run |

All adapters see open answers; answer-frequency and program-length differences remain possible confounds.

## Decision

**Pending.** E3 and E4 are complete for 0 of 1 training seeds; the decision uses all seeds.

Pre-specified rule, applied to the seed-pooled macro CI: Promising if above zero, Not supported if below zero, otherwise Inconclusive; Inconclusive if invalid generations exceed 10%.
