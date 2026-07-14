# Gate A: minutes-prediction accuracy - 20260714T054828Z

Walk-forward WITHIN each of 8 seasons; at gameweek k a model sees only gameweeks 1..k-1 of that player's minutes. Lower is better. Recent-form beating crude here is NECESSARY (a better minutes signal exists) but not sufficient (Gate B decides whether it grows the FPL edge).

## Squad-relevant pool (prior mean minutes >= 45; 56,243 player-gameweeks)

This is the pool that matters - genuine options a manager weighs, where rotation calls are made.

| model | half-life | minutes MAE | p60 Brier | pplay Brier | MAE gain vs crude | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| crude (shipped) | nan | 28.781 | 0.2768 | 0.1753 | +0.000 | - | - |
| recent | 1 | 22.030 | 0.1690 | 0.1209 | +6.750 | 0.0000 | 0.0000 |
| recent | 2 | 23.821 | 0.1670 | 0.1216 | +4.960 | 0.0000 | 0.0000 |
| recent | 3 | 24.935 | 0.1702 | 0.1248 | +3.846 | 0.0000 | 0.0000 |
| recent | 4 | 25.669 | 0.1735 | 0.1278 | +3.112 | 0.0000 | 0.0000 |
| recent | 6 | 26.562 | 0.1787 | 0.1321 | +2.219 | 0.0000 | 0.0000 |
| recent | 10 | 27.404 | 0.1849 | 0.1370 | +1.377 | 0.0000 | 0.0000 |

## All player-gameweeks (>= 5 prior matches; 169,731 obs)

| model | half-life | minutes MAE | p60 Brier | pplay Brier | MAE gain vs crude | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| crude (shipped) | nan | 18.745 | 0.1563 | 0.1487 | +0.000 | - | - |
| recent | 1 | 13.933 | 0.1053 | 0.1085 | +4.812 | 0.0000 | 0.0000 |
| recent | 2 | 15.111 | 0.1039 | 0.1072 | +3.634 | 0.0000 | 0.0000 |
| recent | 3 | 15.885 | 0.1059 | 0.1095 | +2.860 | 0.0000 | 0.0000 |
| recent | 4 | 16.412 | 0.1080 | 0.1118 | +2.333 | 0.0000 | 0.0000 |
| recent | 6 | 17.068 | 0.1114 | 0.1156 | +1.677 | 0.0000 | 0.0000 |
| recent | 10 | 17.696 | 0.1154 | 0.1202 | +1.049 | 0.0000 | 0.0000 |

## Verdict

**Recent-form (half-life 1) predicts minutes better** on the squad pool: MAE 22.030 vs crude 28.781 (+6.750, significant on both tests), and p_60 Brier 0.1690 vs 0.2768. Carry half-life 1 to Gate B - the FPL-edge test that actually decides.