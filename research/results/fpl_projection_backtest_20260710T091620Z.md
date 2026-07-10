# FPL Projection Backtest - 20260710T091620Z

Walk-forward over 33 gameweeks, 26,058 player-gameweeks. To project gameweek k a model sees only gameweeks 1..k-1 and matches played before that gameweek's first kickoff.

**Handicap, stated up front:** FPL publishes injury flags only for the current moment, not historically, so this backtest cannot use them. The live projection does. This therefore *understates* the real system.

## Results

|              |    mae |   rmse |   spearman_all |   spearman_pickable |   top11_points_per_gw |
|:-------------|-------:|-------:|---------------:|--------------------:|----------------------:|
| ours         | 1.0662 | 2.0095 |         0.6843 |              0.1780 |               51.0000 |
| player_ppg   | 1.0474 | 2.0209 |         0.6900 |              0.1312 |               46.5455 |
| player_form5 | 1.0283 | 2.0777 |         0.7215 |              0.1676 |               42.4545 |
| price        | 4.1513 | 4.3895 |         0.4002 |              0.1084 |               38.2424 |
| global_mean  | 1.5025 | 2.3421 |       nan      |            nan      |               21.8182 |

- `spearman_all` — rank correlation across **every** player, including the ~700 fringe players who score 0-2. You never pick from that pool.
- `spearman_pickable` — rank correlation restricted to players averaging 3.0+ points so far (a prediction-time filter, no hindsight). This is the pool a manager actually chooses from.
- `top11_points_per_gw` — pick each model's best 11 and sum what they **really scored**. This is the decision a manager makes, so it is the metric that decides.

## The decisive metric: who picks the better XI?

| baseline | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p |
|---|---|---|---|---|---|---|
| player_ppg | 51.00 | 46.55 | **+4.45** | 22/33 | 0.0520 | 0.0386 |
| player_form5 | 51.00 | 42.45 | **+8.55** | 25/33 | 0.0014 | 0.0022 |
| price | 51.00 | 38.24 | **+12.76** | 26/33 | 0.0001 | 0.0004 |
| global_mean | 51.00 | 21.82 | **+29.18** | 33/33 | 0.0000 | 0.0000 |

## Verdict

**The projection picks a better XI (+4.45 points/GW, +147 across the season, winning 22/33 gameweeks) but the margin is NOT significant on both tests (t p=0.0520, Wilcoxon p=0.0386). With only 33 gameweeks the test has little power: this is suggestive, not proven. One season is not enough evidence to sell it.**

On error metrics the baselines win (MAE: ours 1.0662 vs 1.0474; Spearman over all players: 0.6843 vs 0.6900). That is expected and not a contradiction: those metrics are dominated by hundreds of fringe players who reliably score ~0, which a player's own average predicts almost perfectly. Restricted to the **pickable pool**, the ranking gap reverses in our favour (ours 0.1780 vs 0.1312).

The engine's contribution is fixture information — clean-sheet probability and opponent strength — which changes nothing for a bench player who will score 0 either way, and matters most exactly at the top of the board where picks are made.

## Caveats

- One season (33 scored gameweeks). Low statistical power by construction.
- The top-11 pick ignores FPL's real constraints (budget, max 3 per club, valid formation). All models face the same omission, so the comparison is fair, but the absolute totals are not achievable.
- Injury flags unavailable historically; live they are used, which should favour our projection further.