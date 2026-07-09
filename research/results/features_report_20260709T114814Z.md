# Phase 3a Research Benchmark (tiredness) - 20260709T114814Z

League: EPL (Understat). Walk-forward on 2018-19 to 2025-26 (8 seasons, 3040 matches per model).

**Question:** does the free 'tiredness' signal - days of rest and fixture congestion - improve prediction? Controlled test: an identical logistic model fit on FORM features only vs on FORM + TIREDNESS features. Any out-of-sample gap is the tiredness signal alone.

## Comparison

| model              |   n_predictions |   log_loss |    rps |   brier_score |   ece_home |   ece_draw |   ece_away |   total_runtime_seconds |   mean_fit_seconds_per_fold |
|:-------------------|----------------:|-----------:|-------:|--------------:|-----------:|-----------:|-----------:|------------------------:|----------------------------:|
| ensemble           |       3040.0000 |     0.9925 | 0.2077 |        0.5912 |     0.0313 |     0.0022 |     0.0198 |                 22.6179 |                      2.7068 |
| elo                |       3040.0000 |     0.9956 | 0.2083 |        0.5926 |     0.0373 |     0.0072 |     0.0329 |                  0.4081 |                      0.0498 |
| logistic_form      |       3040.0000 |     1.0207 | 0.2171 |        0.6119 |     0.0180 |     0.0140 |     0.0187 |                  0.6753 |                      0.0838 |
| logistic_form_rest |       3040.0000 |     1.0311 | 0.2178 |        0.6141 |     0.0200 |     0.0175 |     0.0228 |                  1.4151 |                      0.1762 |
| baseline           |       3040.0000 |     1.0673 | 0.2342 |        0.6460 |     0.0107 |     0.0079 |     0.0186 |                  0.0066 |                      0.0008 |

Lower is better for log loss, RPS, Brier score, and ECE.

## Statistical significance (paired, per-match log loss)

| model_a            | model_b            |   n_matches |   mean_log_loss_diff |   paired_t_pvalue |   wilcoxon_pvalue |
|:-------------------|:-------------------|------------:|---------------------:|------------------:|------------------:|
| logistic_form      | logistic_form_rest |        3040 |              -0.0103 |            0.0644 |            0.9925 |
| logistic_form      | baseline           |        3040 |              -0.0466 |            0.0000 |            0.0000 |
| logistic_form      | elo                |        3040 |               0.0251 |            0.0009 |            0.0000 |
| logistic_form      | ensemble           |        3040 |               0.0283 |            0.0001 |            0.0000 |
| logistic_form_rest | baseline           |        3040 |              -0.0362 |            0.0000 |            0.0000 |
| logistic_form_rest | elo                |        3040 |               0.0355 |            0.0002 |            0.0000 |
| logistic_form_rest | ensemble           |        3040 |               0.0386 |            0.0000 |            0.0000 |
| baseline           | elo                |        3040 |               0.0717 |            0.0000 |            0.0000 |
| baseline           | ensemble           |        3040 |               0.0748 |            0.0000 |            0.0000 |
| elo                | ensemble           |        3040 |               0.0031 |            0.0615 |            0.0002 |

## Recommendation

Best model by log loss: **ensemble** (0.9925).
**Tiredness signal (controlled test): adding rest + congestion does NOT help.** form-only log loss 1.0207 vs form+tiredness 1.0311 (not significant: paired t p=0.0644, Wilcoxon p=0.9925).
For context, the current champion (ensemble) scores 0.9925 - these form/tiredness logistic models are diagnostic probes for the tiredness signal, not yet tuned to beat the champion.

## Next steps

The bigger Phase 3 signals - injuries, suspensions and confirmed line-ups - require an API-Football key and become additional builders in research/features, folded into the same logistic/ensemble machinery tested here. Form itself is also a candidate signal to add to the ensemble if it proves additive.

## Calibration plots

- features_calibration_logistic_form_H_20260709T114814Z.png
- features_calibration_logistic_form_D_20260709T114814Z.png
- features_calibration_logistic_form_A_20260709T114814Z.png
- features_calibration_logistic_form_rest_H_20260709T114814Z.png
- features_calibration_logistic_form_rest_D_20260709T114814Z.png
- features_calibration_logistic_form_rest_A_20260709T114814Z.png
- features_calibration_baseline_H_20260709T114814Z.png
- features_calibration_baseline_D_20260709T114814Z.png
- features_calibration_baseline_A_20260709T114814Z.png
- features_calibration_elo_H_20260709T114814Z.png
- features_calibration_elo_D_20260709T114814Z.png
- features_calibration_elo_A_20260709T114814Z.png
- features_calibration_ensemble_H_20260709T114814Z.png
- features_calibration_ensemble_D_20260709T114814Z.png
- features_calibration_ensemble_A_20260709T114814Z.png