# Multi-task VQA pilot: v2_1000train_500test

## Setup

- **Model:** `Qwen/Qwen3.5-4B`, 4-bit NF4 QLoRA; LoRA rank 8, alpha 16, lr 0.0001, effective batch 8, warmup 0.05. Same settings for every experiment and seed.
- **Data:** 1000 train examples per task (one pass), 200 internal-validation and 500 test questions per task. Dataset seed 42; training seeds 42, 43, 44.
- **Equal exposure:** every task contributes the same examples once in every experiment, so E3/E4 adapters take half the optimizer steps of E2.
- **Test:** GQA balanced val split, excluding every question and image of earlier versions. It is not the official GQA test benchmark and was not used to choose any hyperparameter.
- **Routing:** E3/E4 use oracle routing from dataset metadata.

| Split | Questions | Images | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|
| train_pool | 4000 | 3905 | 1000 | 1000 | 1000 | 1000 |
| internal_val | 800 | 774 | 200 | 200 | 200 | 200 |
| test | 2000 | 1836 | 500 | 500 | 500 | 500 |

Excluded from the test pool (v1_400train): 5000 questions and 4883 images, including its 600 previously evaluated test questions.

Test questions share images: 1836 images, up to 4 questions each, 8.3% of images with more than one. CIs therefore use a paired bootstrap over images (all questions of a resampled image move together), 2000 resamples.

Training runs:

| Seed | Run | Tasks | Status | GPU | Step | Wall s | s/step | Peak GiB |
|---|---|---|---|---|---|---|---|---|
| seed42 | e2_joint | T1+T2+T3+T4 | finished | 0 | 500/500 | 4362 | 8.662986 | 4.15 |
| seed42 | e3_query | T1+T2 | finished | 0 | 250/250 | 2010 | 7.94265 | 4.15 |
| seed42 | e3_verify | T3+T4 | finished | 0 | 250/250 | 2059 | 8.137638 | 4.15 |
| seed42 | e4_no_relate | T1+T3 | finished | 0 | 250/250 | 2045 | 8.07715 | 4.15 |
| seed42 | e4_relate | T2+T4 | finished | 0 | 250/250 | 2018 | 7.966397 | 4.15 |
| seed43 | e2_joint | T1+T2+T3+T4 | finished | 2 | 500/500 | 4122 | 8.191565 | 4.15 |
| seed43 | e3_query | T1+T2 | finished | 2 | 250/250 | 2069 | 8.166761 | 4.15 |
| seed43 | e3_verify | T3+T4 | finished | 2 | 250/250 | 2041 | 8.070614 | 4.15 |
| seed43 | e4_no_relate | T1+T3 | finished | 2 | 250/250 | 2041 | 8.065029 | 4.15 |
| seed43 | e4_relate | T2+T4 | finished | 2 | 250/250 | 2080 | 8.227235 | 4.15 |
| seed44 | e2_joint | T1+T2+T3+T4 | finished | 2 | 500/500 | 4082 | 8.11401 | 4.15 |
| seed44 | e3_query | T1+T2 | finished | 2 | 250/250 | 2012 | 7.951189 | 4.15 |
| seed44 | e3_verify | T3+T4 | finished | 2 | 250/250 | 2015 | 7.95784 | 4.15 |
| seed44 | e4_no_relate | T1+T3 | finished | 2 | 250/250 | 2005 | 7.923373 | 4.15 |
| seed44 | e4_relate | T2+T4 | finished | 2 | 250/250 | 2025 | 7.996402 | 4.15 |

## Accuracy

Exact-match accuracy on the new test. Rows from earlier versions use their existing adapters without retraining.

| Run set | Experiment | T1 | T2 | T3 | T4 | Macro | Invalid |
|---|---|---|---|---|---|---|---|
| pretrained | E0 pretrained | 0.650 | 0.472 | 0.828 | 0.800 | 0.688 | 0.000 |
| v1_400train | E3 question-type | 0.686 | 0.568 | 0.874 | 0.872 | 0.750 | 0.000 |
| v1_400train | E4 operation grouping | 0.676 | 0.594 | 0.858 | 0.854 | 0.746 | 0.000 |
| seed42 | E1 single-task | 0.686 | 0.570 | 0.882 | 0.870 | 0.752 | 0.000 |
| seed42 | E2 joint | 0.692 | 0.576 | 0.882 | 0.888 | 0.759 | 0.000 |
| seed42 | E3 question-type | 0.690 | 0.564 | 0.872 | 0.882 | 0.752 | 0.000 |
| seed42 | E4 operation grouping | 0.700 | 0.596 | 0.892 | 0.880 | 0.767 | 0.000 |
| seed43 | E1 single-task | 0.688 | 0.568 | 0.872 | 0.850 | 0.744 | 0.000 |
| seed43 | E2 joint | 0.708 | 0.572 | 0.898 | 0.884 | 0.765 | 0.000 |
| seed43 | E3 question-type | 0.698 | 0.574 | 0.870 | 0.880 | 0.755 | 0.000 |
| seed43 | E4 operation grouping | 0.708 | 0.570 | 0.890 | 0.880 | 0.762 | 0.000 |
| seed44 | E1 single-task | 0.684 | 0.598 | 0.878 | 0.876 | 0.759 | 0.000 |
| seed44 | E2 joint | 0.714 | 0.568 | 0.892 | 0.882 | 0.764 | 0.000 |
| seed44 | E3 question-type | 0.704 | 0.590 | 0.880 | 0.874 | 0.762 | 0.000 |
| seed44 | E4 operation grouping | 0.694 | 0.582 | 0.878 | 0.874 | 0.757 | 0.000 |
| mean ± sd (3 seeds) | E1 single-task | 0.686 ± 0.002 | 0.579 ± 0.017 | 0.877 ± 0.005 | 0.865 ± 0.014 | 0.752 ± 0.007 | 0.000 ± 0.000 |
| mean ± sd (3 seeds) | E2 joint | 0.705 ± 0.011 | 0.572 ± 0.004 | 0.891 ± 0.008 | 0.885 ± 0.003 | 0.763 ± 0.003 | 0.000 ± 0.000 |
| mean ± sd (3 seeds) | E3 question-type | 0.697 ± 0.007 | 0.576 ± 0.013 | 0.874 ± 0.005 | 0.879 ± 0.004 | 0.756 ± 0.005 | 0.000 ± 0.000 |
| mean ± sd (3 seeds) | E4 operation grouping | 0.701 ± 0.007 | 0.583 ± 0.013 | 0.887 ± 0.008 | 0.878 ± 0.003 | 0.762 ± 0.005 | 0.000 ± 0.000 |

## Directed transfer (E1 vs E0)

T[i→j] = Acc_j(E1 adapter trained on i) − Acc_j(E0), mean over 3 seeds. Rows are source tasks, columns target tasks.

| source \ target | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| T1 | +0.036 | +0.088 | +0.047 | +0.046 |
| T2 | +0.031 | +0.107 | +0.041 | +0.046 |
| T3 | +0.031 | +0.082 | +0.049 | +0.069 |
| T4 | +0.035 | +0.069 | +0.040 | +0.065 |

Mean off-diagonal transfer by relationship:

| Relationship | Mean transfer |
|---|---|
| same question type (T1-T2, T3-T4) | +0.057 |
| same operation group (T1-T3, T2-T4) | +0.048 |
| neither (T1-T4, T2-T3) | +0.051 |

Same-operation pairs always cross answer format, so this cannot separate reasoning from format.

## MTL interference (E2 − E1)

I_i = Acc_i(E2 joint) − Acc_i(E1 single-task on i), seed-pooled over 3 seeds, image-clustered 95% CI. Positive: joint training helped.

| Target | I [95% CI] |
|---|---|
| T1 | +0.019 [+0.001, +0.037] |
| T2 | -0.007 [-0.023, +0.009] |
| T3 | +0.013 [-0.001, +0.027] |
| T4 | +0.019 [+0.006, +0.033] |
| Macro (T1-T4) | +0.011 [+0.003, +0.019] |

## E4 − E3

Paired difference on identical test questions, with image-clustered bootstrap 95% CIs. The pooled row averages each question's correctness over seeds first; its CI covers test sampling, and the seed spread below shows training variation.

| Run set | T1 | T2 | T3 | T4 | Macro (T1-T4) | Query macro (T1, T2) | Verify macro (T3, T4) |
|---|---|---|---|---|---|---|---|
| v1_400train | -0.010 [-0.032, +0.012] | +0.026 [+0.004, +0.048] | -0.016 [-0.038, +0.006] | -0.018 [-0.038, +0.000] | -0.004 [-0.016, +0.006] | +0.008 [-0.008, +0.025] | -0.017 [-0.031, -0.003] |
| seed42 | +0.010 [-0.008, +0.030] | +0.032 [+0.008, +0.056] | +0.020 [+0.002, +0.042] | -0.002 [-0.024, +0.020] | +0.015 [+0.004, +0.026] | +0.021 [+0.005, +0.036] | +0.009 [-0.006, +0.024] |
| seed43 | +0.010 [-0.010, +0.030] | -0.004 [-0.030, +0.020] | +0.020 [+0.000, +0.040] | +0.000 [-0.024, +0.022] | +0.007 [-0.004, +0.017] | +0.003 [-0.013, +0.019] | +0.010 [-0.005, +0.025] |
| seed44 | -0.010 [-0.034, +0.014] | -0.008 [-0.030, +0.014] | -0.002 [-0.022, +0.016] | +0.000 [-0.020, +0.020] | -0.005 [-0.016, +0.005] | -0.009 [-0.025, +0.008] | -0.001 [-0.015, +0.014] |
| seeds pooled (3/3) | +0.003 [-0.011, +0.017] | +0.007 [-0.007, +0.020] | +0.013 [+0.001, +0.025] | -0.001 [-0.015, +0.013] | +0.006 [-0.001, +0.012] | +0.005 [-0.005, +0.014] | +0.006 [-0.004, +0.016] |

Per-seed macro E4 − E3: seed42 +0.015, seed43 +0.007, seed44 -0.005 (mean +0.005, sd 0.010; 2/3 positive).

E3 adapters each see one answer format while E4 adapters see both, so this contrast also measures answer-format specialization.

## Decision

**Inconclusive.** E4 minus E3 macro accuracy is +0.006 (95% CI -0.001 to +0.012); the interval includes zero. Improve task design or repeat with more seeds.

Pre-specified rule, applied to the seed-pooled macro CI: Promising if above zero, Not supported if below zero, otherwise Inconclusive; Inconclusive if invalid generations exceed 10%.
