# Quick diagnostic: T2 and grouping

Uses saved predictions on the existing v2 test. A separate fresh holdout inference is reported in `../quick_holdout_25per_task/report.md`.

## E4 − E2

Paired, image-clustered 95% CIs; each question averaged over seeds 42/43/44.

| Target | Difference | 95% CI |
|---|---:|---:|
| T1 | -0.4 pp | [-1.6 pp, +0.8 pp] |
| T2 | +1.1 pp | [-0.4 pp, +2.6 pp] |
| T3 | -0.4 pp | [-1.5 pp, +0.7 pp] |
| T4 | -0.7 pp | [-1.9 pp, +0.6 pp] |
| Macro | -0.1 pp | [-0.7 pp, +0.5 pp] |
| Query | +0.3 pp | [-0.6 pp, +1.3 pp] |
| Verify | -0.5 pp | [-1.4 pp, +0.3 pp] |

## E4 − E3 within answer format

| Format | Difference | 95% CI |
|---|---:|---:|
| Query | +0.5 pp | [-0.5 pp, +1.4 pp] |
| Verify | +0.6 pp | [-0.4 pp, +1.6 pp] |

These are within-format test subsets; E3 and E4 still differed in the answer formats each adapter saw during training.

## Answer frequency in the 4,000 training questions

Frequency uses normalized answer text across all training tasks, matching the exact-match scorer.

| Train frequency | T1 n | T1 E2 | T2 n | T2 E2 |
|---|---:|---:|---:|---:|
| 0 | 21 | 27.0% | 80 | 35.0% |
| 1-2 | 43 | 43.4% | 76 | 39.9% |
| 3-9 | 89 | 68.2% | 183 | 60.8% |
| 10+ | 347 | 77.0% | 161 | 72.3% |

Raw T1 − T2 E2 gap: +13.3 pp (95% image bootstrap [+7.7 pp, +19.2 pp]).
If T1 is reweighted to T2's answer-frequency bins: T1 60.7%, residual gap +3.5 pp (95% image bootstrap [-3.0 pp, +10.0 pp]).
This reweighting is descriptive: answer frequency is associated with task content and does not establish a cause.

## T2 detailed types

| Detailed type | n | E2 accuracy |
|---|---:|---:|
| relO | 151 | 49.0% |
| relS | 144 | 56.2% |
| categoryRelS | 132 | 64.1% |
| directWhich | 31 | 62.4% |

188/500 T2 questions were wrong for E2 in all three seeds.

All intervals condition on these three trained seeds; they do not measure training-seed uncertainty. Subgroup findings are exploratory.
