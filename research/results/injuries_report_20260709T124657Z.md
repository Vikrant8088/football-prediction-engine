# Phase 3b Research Benchmark (injuries) - 20260709T124657Z

League: EPL (Understat matches + API-Football injuries). Walk-forward on 2023-24, 2024-25 (760 matches per model).

**Question:** does knowing how many players each side is missing improve prediction? Controlled test: identical logistic model on FORM vs FORM + INJURY features. Injury data (free API plan) covers 2022/23-2024/25 only, so this is a deliberately small first read.

## Comparison

| model                  |   n_predictions |   log_loss |    rps |   brier_score |   ece_home |   ece_draw |   ece_away |   total_runtime_seconds |   mean_fit_seconds_per_fold |
|:-----------------------|----------------:|-----------:|-------:|--------------:|-----------:|-----------:|-----------:|------------------------:|----------------------------:|
| elo                    |        760.0000 |     0.9789 | 0.2033 |        0.5822 |     0.0333 |     0.0091 |     0.0252 |                  0.1122 |                      0.0551 |
| ensemble               |        760.0000 |     0.9824 | 0.2045 |        0.5845 |     0.0221 |     0.0051 |     0.0258 |                  6.5590 |                      3.1729 |
| logistic_form          |        760.0000 |     1.0268 | 0.2169 |        0.6158 |     0.0473 |     0.0450 |     0.0463 |                  0.0409 |                      0.0196 |
| logistic_form_injuries |        760.0000 |     1.0392 | 0.2196 |        0.6242 |     0.0588 |     0.0554 |     0.0632 |                  0.0696 |                      0.0341 |
| baseline               |        760.0000 |     1.0677 | 0.2346 |        0.6465 |     0.0146 |     0.0039 |     0.0185 |                  0.0019 |                      0.0010 |

Lower is better for log loss, RPS, Brier score, and ECE.

## Statistical significance (paired, per-match log loss)

| model_a                | model_b                |   n_matches |   mean_log_loss_diff |   paired_t_pvalue |   wilcoxon_pvalue |
|:-----------------------|:-----------------------|------------:|---------------------:|------------------:|------------------:|
| logistic_form          | logistic_form_injuries |         760 |              -0.0124 |            0.0558 |            0.1558 |
| logistic_form          | baseline               |         760 |              -0.0409 |            0.0008 |            0.0000 |
| logistic_form          | elo                    |         760 |               0.0479 |            0.0014 |            0.0000 |
| logistic_form          | ensemble               |         760 |               0.0444 |            0.0017 |            0.0001 |
| logistic_form_injuries | baseline               |         760 |              -0.0285 |            0.0383 |            0.0000 |
| logistic_form_injuries | elo                    |         760 |               0.0603 |            0.0003 |            0.0001 |
| logistic_form_injuries | ensemble               |         760 |               0.0568 |            0.0003 |            0.0006 |
| baseline               | elo                    |         760 |               0.0888 |            0.0000 |            0.0000 |
| baseline               | ensemble               |         760 |               0.0853 |            0.0000 |            0.0000 |
| elo                    | ensemble               |         760 |              -0.0035 |            0.2383 |            0.0010 |

## Recommendation

Best model by log loss: **elo** (0.9789).
**Injury signal (controlled test): adding injuries does NOT help.** form-only log loss 1.0268 vs form+injuries 1.0392 (not significant: paired t p=0.0558, Wilcoxon p=0.1558).
Only ~760 evaluation matches (2 seasons - the free plan's injury window), so treat this as a first read, not a settled result. If the signal looks promising, the paid plan (~6 seasons) would let us confirm it and fold injuries into the champion.

## Next steps

If injuries help: (1) upgrade to the paid API plan for ~6 seasons and re-test for significance; (2) weight injuries by player importance (minutes/market value) rather than a raw count; (3) add confirmed line-ups (who actually starts). If they do not help even here, record as rejected - the count-of-absences signal is too blunt without player importance.

## Calibration plots

- injuries_calibration_logistic_form_H_20260709T124657Z.png
- injuries_calibration_logistic_form_D_20260709T124657Z.png
- injuries_calibration_logistic_form_A_20260709T124657Z.png
- injuries_calibration_logistic_form_injuries_H_20260709T124657Z.png
- injuries_calibration_logistic_form_injuries_D_20260709T124657Z.png
- injuries_calibration_logistic_form_injuries_A_20260709T124657Z.png
- injuries_calibration_baseline_H_20260709T124657Z.png
- injuries_calibration_baseline_D_20260709T124657Z.png
- injuries_calibration_baseline_A_20260709T124657Z.png
- injuries_calibration_elo_H_20260709T124657Z.png
- injuries_calibration_elo_D_20260709T124657Z.png
- injuries_calibration_elo_A_20260709T124657Z.png
- injuries_calibration_ensemble_H_20260709T124657Z.png
- injuries_calibration_ensemble_D_20260709T124657Z.png
- injuries_calibration_ensemble_A_20260709T124657Z.png