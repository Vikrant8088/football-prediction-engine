> # ⛔ RETRACTED — DO NOT CITE
>
> **This report's headline result (+4.45 points/GW over `player_ppg`) is wrong.**
> It is kept, unaltered below, as the record of a mistake rather than deleted.
>
> Superseded by `fpl_projection_backtest_multiseason_*.md`, which replays four
> seasons instead of one. Three defects, all of which flattered us:
>
> 1. **The XI was illegal.** Ranking players by projected points and taking the
>    top 11 fielded **5.9 goalkeepers** on average; FPL permits one. Clean-sheet
>    and save points make keepers look efficient in isolation, so the metric
>    rewarded the model that rated them highest — ours.
> 2. **Double gameweeks double-counted.** A player with two fixtures appeared
>    twice and could be picked twice in one XI.
> 3. **The `price` baseline used end-of-season prices**, contaminated by the very
>    season it was predicting.
>
> On the corrected metric across 131 gameweeks the gain is **+1.83 pts/GW,
> p=0.13 — not significant.** The single-season result was noise plus artifact.
>
> Why one season could not have caught this: 2025/26's new defensive-contribution
> rule lifts outfielders enough that goalkeepers drop out of the naive XI by
> themselves (0.4 per XI, versus 5.9 in 2022/23). The bug was hiding behind a
> rule change.

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