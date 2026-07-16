# Gate B (Phase 6e): promoted-team Elo prior on the FPL edge - 20260716T051341Z

Pre-registered primary: £100m squad + captain, 8 Understat-xG seasons (263 GW), head-to-head vs the CHAMPION (recent-form minutes, no prior), which beats player_ppg by **+7.97 pts/GW**. Only the Elo promoted prior varies.

## Each configuration's own edge over player_ppg

| config | gain/GW vs ppg | t p | Wilcoxon p |
|---|---|---|---|
| champion (no prior) | **+7.97** | 0.0000 | 0.0000 |
| promoted prior = 100 | **+7.94** | 0.0000 | 0.0000 |
| promoted prior = 150 | **+7.92** | 0.0000 | 0.0000 |

## The decisive test: challenger vs champion, head-to-head

| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | Holm survives? | non-DC gain/GW | ships? |
|---|---|---|---|---|---|---|---|
| promoted prior = 100 | **+0.004** | 42/45 | 0.9885 | 0.7732 | no | -0.096 | no |
| promoted prior = 150 | **-0.023** | 48/53 | 0.9342 | 0.6989 | no | -0.030 | no |

## Per-season replication (head-to-head vs champion)

| season | promoted prior = 100 | promoted prior = 150 |
|---|---|---|
| 2018-19 | +0.455 | +0.030 |
| 2019-20 | -0.152 | +0.091 |
| 2020-21 | +0.152 | +0.424 |
| 2021-22 | +0.030 | +0.182 |
| 2022-23 | +0.594 | +0.969 |
| 2023-24 | -1.152 | -1.182 |
| 2024-25 | -0.576 | -0.697 |
| 2025-26 | +0.697 | +0.030 |

## Verdict

**No challenger clears the bar.** Best arm (promoted prior = 100) moves the edge by +0.004 pts/GW head-to-head (t p=0.9885, Wilcoxon p=0.7732) - not significant. Getting promoted teams' strength right is more correct and improved their fixture prediction (Gate A), but by the scored range the weekly-refit model has enough of their results that the prior barely changes the squad decision. Shipped cold-start stands; recorded as a measured null.

**Robustness of the best arm:** positive in 5/8 seasons; pooled +0.004/GW, dropping its best season (2025-26) leaves -0.096/GW.

## Caveats

- Head-to-head isolates ONLY the Elo prior: identical minutes model, goal models, optimizer, budget, captain and baseline.
- Penalties 100/150 carried from Gate A, not tuned on the edge; Holm across the two.