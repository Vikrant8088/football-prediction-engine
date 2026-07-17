# The minutes ceiling (fast, unbudgeted legal XI) — 20260717T052438Z

Perfect minutes LEAK each gameweek's actual minutes: an upper bound on any lineup/minutes signal, not a shippable model. Scored on the greedy legal XI (no budget, no captain) because the £100m squad solve is pathologically slow on finely-spaced perfect-minutes projections. **Directional, not the pre-registered £100m + captain figure.**

263 gameweeks, 8 seasons.

| picker | actual pts/GW | edge vs `player_ppg` |
|---|---|---|
| `player_ppg` baseline | 50.61 | — |
| recent-form minutes (shipped champion) | 55.62 | **+5.01/GW** |
| PERFECT minutes (ceiling) | 60.22 | **+9.60/GW** |

## The headroom (perfect − recent-form)

> **+4.593 pts/GW** (t p=0.0000, Wilcoxon p=0.0000), won 158/263 gameweeks.

## Per-season headroom

| season | perfect − recent-form (pts/GW) | GWs won |
|---|---|---|
| 2018-19 | +2.939 | 18/33 |
| 2019-20 | +2.667 | 19/33 |
| 2020-21 | +6.424 | 26/33 |
| 2021-22 | +12.667 | 23/33 |
| 2022-23 | +3.469 | 18/32 |
| 2023-24 | +1.939 | 18/33 |
| 2024-25 | +1.606 | 16/33 |
| 2025-26 | +5.000 | 20/33 |

## Reading it

This headroom is the MAXIMUM any minutes/lineup data could add on top of the recent-form model. Live predicted lineups are imperfect, so they capture only a FRACTION of it.

**Positive in 8/8 seasons.**

**Important caveat — the live system is already better informed than this champion.** The backtest cannot see FPL's injury flags (they are published only for the current moment), so the champion here is blind to availability, while the LIVE projection already scales minutes by `chance_of_playing`. Part of this headroom is therefore already captured live. The remaining prize for predicted lineups is the part flags cannot give: **rotation** — who a manager actually picks among fit players.