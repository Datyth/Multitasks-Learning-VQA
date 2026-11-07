# Quick holdout evaluation

100 fresh GQA balanced-val questions: 25 per task, 100 distinct images, no question or image overlap with v1/v2 selections. Existing seed-42 adapters were used without retraining.

| Model | T1 | T2 | T3 | T4 | Macro |
|---|---:|---:|---:|---:|---:|
| E2 | 72% | 48% | 100% | 88% | 77% |
| E3 | 72% | 40% | 100% | 84% | 74% |
| E4 | 76% | 48% | 100% | 88% | 78% |

Paired 95% bootstrap CIs over the 100 distinct test images. Each task has only 25 questions.

| Difference | Macro | 95% CI | Query | Verify |
|---|---:|---:|---:|---:|
| E4-E2 | +1.0 pp | [-2.4, +4.5] pp | +2.0 pp | +0.0 pp |
| E4-E3 | +4.0 pp | [+0.8, +8.3] pp | +6.0 pp | +2.0 pp |

## Paired disagreements

Wins/losses are questions where the first model is correct and the second is wrong / vice versa.

| Comparison | T1 wins/losses | T2 | T3 | T4 | Exact two-sided p |
|---|---:|---:|---:|---:|---:|
| E4-E2 | 1/0 | 1/1 | 0/0 | 0/0 | 1.000 |
| E4-E3 | 1/0 | 2/0 | 0/0 | 1/0 | 0.125 |

In T2, 13/25 answers appeared at most twice in the 4,000 training questions (normalized exact text).

With only four E4/E3 discordant questions, the percentile bootstrap CI is optimistic; the exact paired p-value is more cautious.
This is a small pilot with one training seed; its intervals cover test-image sampling only. E3/E4 also differ in the answer formats each adapter saw during training.
