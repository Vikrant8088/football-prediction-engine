# Phase 6a Stage 2: window/decay on the FPL pts/GW edge - 20260713T151329Z

Pre-registered primary endpoint: the GBP100m squad + captain, 8 Understat-xG seasons (263 GW). The shipped team model (Dixon-Coles xi=0.0065, expanding window) already beats player_ppg by **+5.02 pts/GW** here (t p=0.0000, Wilcoxon p=0.0000) - this is the Phase 5e headline, reproduced as the baseline to improve on.

The question: does a gentler decay (and/or a bounded window) grow that edge? The decisive column is the paired **head-to-head** - challenger squad points minus default squad points, gameweek by gameweek. A challenger ships only if it is positive and significant on BOTH tests after Holm correction across the two challengers, and does not turn negative once the 2025/26 defensive-contribution season is removed.

## Each configuration's own edge over player_ppg

| config | gain/GW vs ppg | t p | Wilcoxon p |
|---|---|---|---|
| default (xi=0.0065, expanding) | **+5.02** | 0.0000 | 0.0000 |
| E1 aggressive (xi=0.012, expanding) | **+5.33** | 0.0000 | 0.0000 |
| E2 very aggressive (xi=0.02, expanding) | **+5.51** | 0.0000 | 0.0000 |

## The decisive test: challenger vs default, head-to-head

| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | Holm survives? | non-DC gain/GW | ships? |
|---|---|---|---|---|---|---|---|
| E1 aggressive (xi=0.012, expanding) | **+0.335** | 67/62 | 0.3579 | 0.3969 | no | +0.487 | no |
| E2 very aggressive (xi=0.02, expanding) | **+0.517** | 95/83 | 0.2961 | 0.4227 | no | +0.704 | no |

## Per-season replication (head-to-head vs default)

| season | E1 aggressive | E2 very aggressive |
|---|---|---|
| 2018-19 | -0.394 | +0.273 |
| 2019-20 | +1.424 | +2.455 |
| 2020-21 | +0.848 | +0.848 |
| 2021-22 | -0.606 | -0.212 |
| 2022-23 | +0.406 | -0.406 |
| 2023-24 | +0.788 | +1.121 |
| 2024-25 | +0.939 | +0.818 |
| 2025-26 | -0.727 | -0.788 |

## Verdict

**No challenger clears the bar.** The gentler decay improves the goal models' own clean-sheet and 1X2 scoring (Stage 1), but that improvement does NOT convert into a larger squad-points edge once the grid is rescaled to the Elo-led ensemble marginals and pushed through the optimizer - exactly the dampening flagged before this stage. The shipped xi=0.0065 / expanding-window default stands; the window/decay lever is recorded as a measured null on the FPL endpoint, consistent with the low predictability ceiling.

## Caveats

- Head-to-head isolates ONLY the team-model change: both arms share the identical player rates, optimizer, budget, captain rule and player_ppg baseline, so any difference is the decay/window and nothing else.
- Squad-cache tags are distinct per frame, so no arm reuses another's squads (the collision that once faked a perfect result).
- The DC clean-sheet gain failed Wilcoxon at Stage 1; a null here is the expected, not a surprising, outcome.