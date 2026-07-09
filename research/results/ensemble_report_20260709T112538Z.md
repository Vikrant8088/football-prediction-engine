# Phase 2 Research Benchmark (ensemble) - 20260709T112538Z

League: EPL (Understat). Evaluated (walk-forward, expanding window) on 2018-19 to 2025-26 (8 seasons, 3040 matches per model).

**Question:** does a walk-forward-weighted ensemble of the strong, diverse base models (Elo + Poisson-xG + Dixon-Coles-xG) beat the best single model, Elo? The ensemble fits its weights only on an inner temporal split of each fold's training data - it never sees the evaluated season.

## Comparison

| model          |   n_predictions |   log_loss |    rps |   brier_score |   ece_home |   ece_draw |   ece_away |   total_runtime_seconds |   mean_fit_seconds_per_fold |
|:---------------|----------------:|-----------:|-------:|--------------:|-----------:|-----------:|-----------:|------------------------:|----------------------------:|
| ensemble       |       3040.0000 |     0.9925 | 0.2077 |        0.5912 |     0.0313 |     0.0022 |     0.0198 |                 24.2368 |                      2.9168 |
| elo            |       3040.0000 |     0.9956 | 0.2083 |        0.5926 |     0.0373 |     0.0072 |     0.0329 |                  0.3817 |                      0.0467 |
| dixon_coles_xg |       3040.0000 |     1.0107 | 0.2130 |        0.6031 |     0.0264 |     0.0071 |     0.0287 |                 11.0708 |                      1.3046 |
| poisson_xg     |       3040.0000 |     1.0198 | 0.2171 |        0.6101 |     0.0322 |     0.0093 |     0.0248 |                  2.7238 |                      0.2901 |
| baseline       |       3040.0000 |     1.0673 | 0.2342 |        0.6460 |     0.0107 |     0.0079 |     0.0186 |                  0.0065 |                      0.0008 |

Lower is better for log loss, RPS, Brier score, and ECE.

## Statistical significance (paired, per-match log loss)

| model_a        | model_b        |   n_matches |   mean_log_loss_diff |   paired_t_pvalue |   wilcoxon_pvalue |
|:---------------|:---------------|------------:|---------------------:|------------------:|------------------:|
| baseline       | elo            |        3040 |               0.0717 |            0.0000 |            0.0000 |
| baseline       | poisson_xg     |        3040 |               0.0475 |            0.0000 |            0.0000 |
| baseline       | dixon_coles_xg |        3040 |               0.0566 |            0.0000 |            0.0000 |
| baseline       | ensemble       |        3040 |               0.0748 |            0.0000 |            0.0000 |
| elo            | poisson_xg     |        3040 |              -0.0242 |            0.0000 |            0.0000 |
| elo            | dixon_coles_xg |        3040 |              -0.0151 |            0.0082 |            0.0000 |
| elo            | ensemble       |        3040 |               0.0031 |            0.0615 |            0.0002 |
| poisson_xg     | dixon_coles_xg |        3040 |               0.0091 |            0.0555 |            0.0036 |
| poisson_xg     | ensemble       |        3040 |               0.0273 |            0.0000 |            0.0000 |
| dixon_coles_xg | ensemble       |        3040 |               0.0182 |            0.0001 |            0.0000 |

## Recommendation

Best model by log loss: **ensemble** (0.9925).
Ensemble weights (averaged across walk-forward folds): elo 72%, poisson_xg 12%, dixon_coles_xg 16%.
**The ensemble beats Elo** (0.9925 vs 0.9956) (not significant: paired t p=0.0615, Wilcoxon p=0.0002) - the first model to dethrone Elo. Blending the rating and scoreline views does better than either alone. New champion: ensemble.

## Next steps

If the ensemble wins, it becomes the model to beat and the next gains come from adding genuinely new signals (Phase 3: injuries, lineups, rest) and an ML base model on a feature store. A remaining lever for any of the current models is hyperparameter tuning (Elo K / home advantage; Dixon-Coles xi), still on literature defaults.

## Calibration plots

- ensemble_calibration_baseline_H_20260709T112538Z.png
- ensemble_calibration_baseline_D_20260709T112538Z.png
- ensemble_calibration_baseline_A_20260709T112538Z.png
- ensemble_calibration_elo_H_20260709T112538Z.png
- ensemble_calibration_elo_D_20260709T112538Z.png
- ensemble_calibration_elo_A_20260709T112538Z.png
- ensemble_calibration_poisson_xg_H_20260709T112538Z.png
- ensemble_calibration_poisson_xg_D_20260709T112538Z.png
- ensemble_calibration_poisson_xg_A_20260709T112538Z.png
- ensemble_calibration_dixon_coles_xg_H_20260709T112538Z.png
- ensemble_calibration_dixon_coles_xg_D_20260709T112538Z.png
- ensemble_calibration_dixon_coles_xg_A_20260709T112538Z.png
- ensemble_calibration_ensemble_H_20260709T112538Z.png
- ensemble_calibration_ensemble_D_20260709T112538Z.png
- ensemble_calibration_ensemble_A_20260709T112538Z.png