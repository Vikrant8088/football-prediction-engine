# Multi-League Benchmark - 20260710T052635Z

Europe's top five leagues (Understat), each trained and evaluated strictly within itself, walk-forward, 4 training-only seasons then 8 evaluated. Pooled scoring across 14284 matches per model.

**Questions:** (1) does the champion established on the Premier League generalize to other leagues, or was it an artefact of one competition? (2) Pooling ~5x the matches, is the ensemble's borderline win over Elo real?

## Log loss by league (lower is better)

|                |    EPL |   La_liga |   Bundesliga |   Serie_A |   Ligue_1 |   pooled |
|:---------------|-------:|----------:|-------------:|----------:|----------:|---------:|
| baseline       | 1.0673 |    1.0707 |       1.0736 |    1.0842 |    1.0763 |   1.0744 |
| dixon_coles_xg | 1.0107 |    1.0023 |       1.0089 |    1.0240 |    1.0267 |   1.0145 |
| elo            | 0.9956 |    0.9995 |       1.0064 |    0.9971 |    1.0333 |   1.0058 |
| ensemble       | 0.9925 |    0.9945 |       0.9988 |    0.9963 |    1.0212 |   1.0003 |
| poisson_xg     | 1.0198 |    0.9968 |       0.9985 |    1.0099 |    1.0224 |   1.0096 |

## Pooled comparison (all leagues)

| model          |   n_predictions |   log_loss |    rps |   brier_score |   ece_home |   ece_draw |   ece_away |   total_runtime_seconds |   mean_fit_seconds_per_fold |
|:---------------|----------------:|-----------:|-------:|--------------:|-----------:|-----------:|-----------:|------------------------:|----------------------------:|
| ensemble       |      14284.0000 |     1.0003 | 0.2049 |        0.5971 |     0.0191 |     0.0124 |     0.0077 |                380.1388 |                      9.0920 |
| elo            |      14284.0000 |     1.0058 | 0.2065 |        0.6005 |     0.0304 |     0.0107 |     0.0185 |                  6.7686 |                      0.1663 |
| poisson_xg     |      14284.0000 |     1.0096 | 0.2079 |        0.6035 |     0.0198 |     0.0135 |     0.0075 |                 48.5167 |                      1.0174 |
| dixon_coles_xg |      14284.0000 |     1.0145 | 0.2093 |        0.6067 |     0.0171 |     0.0127 |     0.0123 |                160.1861 |                      3.7783 |
| baseline       |      14284.0000 |     1.0744 | 0.2306 |        0.6503 |     0.0157 |     0.0030 |     0.0126 |                  0.0855 |                      0.0021 |

## Pooled statistical significance (paired, per-match log loss)

| model_a        | model_b        |   n_matches |   mean_log_loss_diff |   paired_t_pvalue |   wilcoxon_pvalue |
|:---------------|:---------------|------------:|---------------------:|------------------:|------------------:|
| baseline       | elo            |       14284 |               0.0687 |            0.0000 |            0.0000 |
| baseline       | poisson_xg     |       14284 |               0.0648 |            0.0000 |            0.0000 |
| baseline       | dixon_coles_xg |       14284 |               0.0600 |            0.0000 |            0.0000 |
| baseline       | ensemble       |       14284 |               0.0741 |            0.0000 |            0.0000 |
| elo            | poisson_xg     |       14284 |              -0.0039 |            0.0572 |            0.0000 |
| elo            | dixon_coles_xg |       14284 |              -0.0087 |            0.0001 |            0.0000 |
| elo            | ensemble       |       14284 |               0.0055 |            0.0000 |            0.0000 |
| poisson_xg     | dixon_coles_xg |       14284 |              -0.0048 |            0.0089 |            0.5609 |
| poisson_xg     | ensemble       |       14284 |               0.0094 |            0.0000 |            0.0000 |
| dixon_coles_xg | ensemble       |       14284 |               0.0142 |            0.0000 |            0.0000 |

## Recommendation

**Generalization:** the ensemble is the best model in **4 of 5** leagues.

Winner per league: EPL -> ensemble, La_liga -> ensemble, Bundesliga -> poisson_xg, Serie_A -> ensemble, Ligue_1 -> ensemble.

**Pooled (all 5 leagues, 14284 matches per model):** best is **ensemble** (1.0003).

Pooled head-to-heads (the point of pooling is statistical power the single-league test lacked):
- **ensemble** 1.0003 vs **elo** 1.0058 -> ensemble better (SIGNIFICANT: paired t p=0.0000, Wilcoxon p=0.0000).
- **ensemble** 1.0003 vs **dixon_coles_xg** 1.0145 -> ensemble better (SIGNIFICANT: paired t p=0.0000, Wilcoxon p=0.0000).
- **elo** 1.0058 vs **dixon_coles_xg** 1.0145 -> elo better (SIGNIFICANT: paired t p=0.0001, Wilcoxon p=0.0000).

**Verdict: the ensemble's win over Elo is now significant on both tests.** With ~5x the matches, the margin that was borderline on the EPL alone holds up. The ensemble is the champion, and it generalizes.

## Notes

Each league is trained only on its own history - no cross-league transfer. Pooling happens at scoring time only. Match keys are namespaced by league so fixtures cannot collide in the paired tests.

## Calibration plots (pooled)

- multileague_calibration_baseline_H_20260710T052635Z.png
- multileague_calibration_baseline_D_20260710T052635Z.png
- multileague_calibration_baseline_A_20260710T052635Z.png
- multileague_calibration_elo_H_20260710T052635Z.png
- multileague_calibration_elo_D_20260710T052635Z.png
- multileague_calibration_elo_A_20260710T052635Z.png
- multileague_calibration_poisson_xg_H_20260710T052635Z.png
- multileague_calibration_poisson_xg_D_20260710T052635Z.png
- multileague_calibration_poisson_xg_A_20260710T052635Z.png
- multileague_calibration_dixon_coles_xg_H_20260710T052635Z.png
- multileague_calibration_dixon_coles_xg_D_20260710T052635Z.png
- multileague_calibration_dixon_coles_xg_A_20260710T052635Z.png
- multileague_calibration_ensemble_H_20260710T052635Z.png
- multileague_calibration_ensemble_D_20260710T052635Z.png
- multileague_calibration_ensemble_A_20260710T052635Z.png