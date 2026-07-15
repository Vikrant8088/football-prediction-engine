# Gate B (Phase 6c): penalty split on the FPL pts/GW edge - 20260715T105216Z

Pre-registered primary: £100m squad + captain, 8 Understat-xG seasons (263 GW), head-to-head vs the CHAMPION (recent-form minutes, penalties lumped), which already beats player_ppg by **+7.97 pts/GW**. Only the penalty split varies.

## Each configuration's own edge over player_ppg

| config | gain/GW vs ppg | t p | Wilcoxon p |
|---|---|---|---|
| champion (penalties lumped) | **+7.97** | 0.0000 | 0.0000 |
| penalty split (open-play scaled, penalties flat) | **+8.10** | 0.0000 | 0.0000 |

## The decisive test: challenger vs champion, head-to-head

| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | Holm survives? | non-DC gain/GW | ships? |
|---|---|---|---|---|---|---|---|
| penalty split (open-play scaled, penalties flat) | **+0.163** | 49/45 | 0.6002 | 0.5271 | no | +0.196 | no |

## Per-season replication (head-to-head vs champion)

| season | h2h gain/GW |
|---|---|
| 2018-19 | -0.455 |
| 2019-20 | +0.788 |
| 2020-21 | +0.333 |
| 2021-22 | +0.152 |
| 2022-23 | +1.781 |
| 2023-24 | -1.152 |
| 2024-25 | -0.030 |
| 2025-26 | -0.061 |

## Verdict

**The penalty split does not clear the bar.** It moves the edge by +0.163 pts/GW head-to-head (t p=0.6002, Wilcoxon p=0.5271) - not significant. Splitting penalties from open-play xG is more principled and marginally changes premium takers' projections, but the correction is too small to move the squad decision measurably. The lumped model stands; recorded as a measured null (consistent with the research finding no verified penalty effect and with xG already capturing takers).

**Robustness of the best arm:** positive in 4/8 seasons; pooled +0.163/GW, dropping its best season (2022-23) leaves -0.061/GW.

## Caveats

- Head-to-head isolates ONLY the penalty split: identical minutes model, team model, optimizer, budget, captain and baseline.
- penalty_multiplier=1.0 is pre-specified (penalties fixture-independent), not tuned on the edge.
- Backtest handicap unchanged: no live penalty-taker feed; the split is inferred from each player's own realised penalty xG.