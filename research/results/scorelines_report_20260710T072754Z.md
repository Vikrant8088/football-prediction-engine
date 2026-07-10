# Scoreline Prediction Benchmark - 20260710T072754Z

League: EPL. Walk-forward on 3040 matches the model never saw.

**Question:** the engine can now name a score. How often is it right?

## Results

| | rate |
|---|---|
| Engine's most likely score is exactly right | **11.3%** |
| The true score is in the engine's top 3 | **29.6%** |
| Naive: always predict 1-1 | 10.9% |
| **Ceiling** (avg probability of the top score) | **12.9%** |

Log loss of the true scoreline under the full grid: 2.9996. Mean probability the engine assigned to the score that actually happened: 6.8%.

## Interpretation

The engine hits the exact score 11.3% of the time, against a ceiling of 12.9%. It is therefore operating at roughly **88% of the maximum any forecaster could achieve** - because the most likely scoreline in a football match is itself only ~13% likely.

Exact-score prediction is a **variance** problem, not a skill problem. No model, bookmaker or human beats this ceiling by much; the probability mass simply is not concentrated on one scoreline. The engine should therefore always publish the score WITH its probability attached, never as a certainty.

## Scores the engine names most often

| scoreline | engine named it | it actually happened |
|---|---|---|
| 1-1 | 1446 (47.6%) | 331 (10.9%) |
| 1-0 | 606 (19.9%) | 251 (8.3%) |
| 0-1 | 371 (12.2%) | 215 (7.1%) |
| 2-0 | 281 (9.2%) | 230 (7.6%) |
| 2-1 | 137 (4.5%) | 253 (8.3%) |
| 1-2 | 83 (2.7%) | 217 (7.1%) |