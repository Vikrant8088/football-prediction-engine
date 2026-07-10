# FPL Projection Backtest (multi-season) - 20260710T101328Z

Walk-forward across **4 seasons**, 131 gameweeks, 96,303 player-gameweeks. To project gameweek k a model sees only gameweeks 1..k-1 of that season and matches played before that gameweek's first kickoff.

Seasons: 2022-23, 2023-24, 2024-25, 2025-26. Earlier seasons are excluded because FPL published no xG before 2022/23, and replaying them on realised goals would test a different model.

**Handicap:** FPL publishes injury flags only for the current moment, not historically, so this backtest cannot use them. The live projection does. This therefore *understates* the real system.

## Pooled results

|              |    mae |   rmse |   spearman_all |   spearman_pickable |   legal_xi_points_per_gw |   naive_top11_points_per_gw |   naive_xi_goalkeepers |
|:-------------|-------:|-------:|---------------:|--------------------:|-------------------------:|----------------------------:|-----------------------:|
| ours         | 1.1023 | 2.0717 |         0.6564 |              0.1808 |                  52.3435 |                     54.4275 |                 1.7252 |
| player_ppg   | 1.0770 | 2.0852 |         0.6666 |              0.1367 |                  50.5115 |                     54.1832 |                 0.1145 |
| player_form5 | 1.0649 | 2.1421 |         0.6995 |              0.1879 |                  46.1527 |                     48.5802 |                 0.3664 |
| price        | 4.1492 | 4.3744 |         0.3944 |              0.0947 |                  43.4351 |                     45.4580 |                 0.0000 |
| global_mean  | 1.5121 | 2.4028 |       nan      |            nan      |                  15.6641 |                      7.2977 |                 1.1985 |

- `spearman_all` - rank correlation across **every** player, including the ~700 fringe players who score 0-2. You never pick from that pool.
- `spearman_pickable` - rank correlation restricted to players averaging 3.0+ points so far (a prediction-time filter, no hindsight). This is the pool a manager actually chooses from.
- `legal_xi_points_per_gw` - **the metric that decides.** Field the best XI this model could actually field (legal formation, max 3 per club) and sum what those players really scored.
- `naive_top11_points_per_gw` / `naive_xi_goalkeepers` - the OLD, invalid metric and its tell. Ranking by projected points and taking the top 11 fields multiple goalkeepers; a manager fields one. Reported only to document why the earlier single-season result was wrong.

## The decisive metric: who picks the better XI?

| baseline | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p | both? |
|---|---|---|---|---|---|---|---|
| player_ppg | 52.34 | 50.51 | **+1.83** | 72/131 | 0.1303 | 0.0913 | NO |
| player_form5 | 52.34 | 46.15 | **+6.19** | 86/131 | 0.0000 | 0.0000 | yes |
| price | 52.34 | 43.44 | **+8.91** | 94/131 | 0.0000 | 0.0000 | yes |
| global_mean | 52.34 | 15.66 | **+36.68** | 131/131 | 0.0000 | 0.0000 | yes |

## Replication: ours vs `player_ppg`, season by season

One pooled p-value can be driven by a single lucky season. These are independent replications.

| season | our XI | ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |
|---|---|---|---|---|---|---|---|
| 2022-23 | 52.03 | 54.25 | **-2.22** | 13/32 | 0.3853 | 0.3467 | NO |
| 2023-24 | 53.33 | 51.73 | **+1.61** | 16/33 | 0.4692 | 0.5873 | NO |
| 2024-25 | 50.82 | 49.42 | **+1.39** | 20/33 | 0.5802 | 0.1872 | NO |
| 2025-26 | 53.18 | 46.76 | **+6.42** | 23/33 | 0.0082 | 0.0166 | yes |

## Verdict

**The projection picks a better XI (+1.83 points/GW, winning 72/131 gameweeks) but the margin is NOT significant on both tests (t p=0.1303, Wilcoxon p=0.0913). Suggestive, not proven.** The edge appears in **3/4 seasons** independently (significant on both tests in 1/4).

On error metrics the baselines win (MAE: ours 1.1023 vs 1.0770; Spearman over all players: 0.6564 vs 0.6666). That is expected and not a contradiction: those metrics are dominated by hundreds of fringe players who reliably score ~0, which a player's own average predicts almost perfectly. Restricted to the **pickable pool**, the ranking gap reverses in our favour (ours 0.1808 vs 0.1367).

The engine's contribution is fixture information - clean-sheet probability and opponent strength - which changes nothing for a bench player who will score 0 either way, and matters most exactly at the top of the board where picks are made.

## Caveats

- The XI respects formation and the max-3-per-club rule but still ignores **budget**, and there is no captain (FPL doubles one player's score). All models face the same omission, so the comparison is fair.
- Within a formation the XI is filled greedily by projected points subject to the club cap, not solved exactly. Applied identically to every model.
- Injury flags unavailable historically; live they are used, which should favour our projection further.
- `defensive_contribution` exists only in 2025/26; earlier seasons neither award nor project those points.
- Double gameweeks are aggregated to one row per player-gameweek, so a player cannot be picked twice in one XI.
- `price` is the price at that gameweek, not end-of-season.

## Data source

Per-gameweek history from the community FPL archive (https://github.com/vaastav/Fantasy-Premier-League, CC BY 4.0), which mirrors FPL's own `element-summary` endpoint. Team model trained on Understat match + xG data.