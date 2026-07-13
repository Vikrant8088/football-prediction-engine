# FPL: the real game — a £100m squad, not a wish list — 20260713T112911Z

Walk-forward across **8 seasons, 263 gameweeks, 174,517 player-gameweeks**. Each gameweek every model buys a legal **15-man squad** (2 GKP, 5 DEF, 5 MID, 3 FWD; max 3 per club across all fifteen; £100m), starts its best legal eleven, and scores only those eleven.

The squad is chosen by branch and bound, **verified against exhaustive enumeration** of every legal XI and every legal bench behind it. Every model gets the same optimizer, so a difference here is a difference in the **projections**, not in how the team was assembled.

> **Why not an "XI budget"?** There isn't one. The bench costs real money and its shape is forced by the formation (the squad quota is fixed), so the money left for the eleven is £83.1m–£84.3m depending on what you play. An earlier version of this benchmark swept flat XI budgets up to £90m — squads that cannot be bought — and £90m gave the best result in the sweep. Deleted.

## Primary endpoint (pre-specified)

The game as played: **£100m squad, with a captain.** Named before any of these numbers existed. A single pre-specified endpoint needs no multiplicity correction — and it is not the flattering choice, because captaincy *reduces* our measured gain.

> **+4.99 points per gameweek** (59.32 vs 54.32), winning 165/263 gameweeks, +190 across a 38-gameweek season.
>
> paired t p=0.0000 · Wilcoxon p=0.0000 → **PASSES** the two-test rule.

## Sensitivity: a poorer squad

A richer squad is not a variation — £100m is the rule. A poorer one is.

| squad budget | captain | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p | both? | **corrected** |
|---|---|---|---|---|---|---|---|---|---|
| £100m | no | 52.47 | 48.14 | **+4.33** | 159/263 | 0.0000 | 0.0000 | ✅ | **✅** |
| £100m | yes | 59.32 | 54.32 | **+4.99** | 165/263 | 0.0000 | 0.0000 | ✅ | **✅** |
| £95m | no | 51.11 | 46.63 | **+4.48** | 155/263 | 0.0000 | 0.0000 | ✅ | **✅** |
| £95m | yes | 58.01 | 52.85 | **+5.16** | 160/263 | 0.0000 | 0.0000 | ✅ | **✅** |
| £90m | no | 48.81 | 44.43 | **+4.38** | 161/263 | 0.0000 | 0.0000 | ✅ | **✅** |
| £90m | yes | 55.75 | 50.24 | **+5.51** | 159/263 | 0.0000 | 0.0000 | ✅ | **✅** |

**Multiplicity.** 6 configurations were tested, so reporting whichever clears p<0.05 would be cherry-picking. `corrected` applies **Holm-Bonferroni** across all 6, to both tests: **6/6 survive.** Written before these numbers existed, and conservative here because the cells are heavily correlated.

## Replication: season by season (primary configuration)

One pooled p-value carried by a single lucky season is what invalidated the earlier claim. These are independent replications.

| season | our XI | ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |
|---|---|---|---|---|---|---|---|
| 2018-19 | 62.45 | 52.24 | **+10.21** | 26/33 | 0.0005 | 0.0012 | ✅ |
| 2019-20 | 53.85 | 52.12 | **+1.73** | 18/33 | 0.6131 | 0.6678 | ❌ |
| 2020-21 | 55.64 | 50.30 | **+5.33** | 22/33 | 0.0385 | 0.0204 | ✅ |
| 2021-22 | 65.21 | 55.24 | **+9.97** | 25/33 | 0.0068 | 0.0103 | ✅ |
| 2022-23 | 62.66 | 59.16 | **+3.50** | 16/32 | 0.2915 | 0.3439 | ❌ |
| 2023-24 | 55.15 | 54.55 | **+0.61** | 14/33 | 0.7849 | 0.5873 | ❌ |
| 2024-25 | 61.48 | 57.94 | **+3.55** | 24/33 | 0.2515 | 0.1502 | ❌ |
| 2025-26 | 58.18 | 53.18 | **+5.00** | 20/33 | 0.0829 | 0.0777 | ❌ |

## The load-bearing check: does the edge survive without the new rule?

2025/26 is the only season with the **defensive-contribution rule**, which we model and the baselines do not — in the projection-only backtest it was ~60% of a *perishable* edge. So the test that matters: drop 2025/26 and re-measure on the seasons that lack the rule.

> **230 non-DC gameweeks, £100m + captain:** +4.99 pts/GW, winning 145/230, t p=0.0000, Wilcoxon p=0.0000 → **still significant.**

The edge does **not** depend on the new rule: with 2025/26 removed it is +4.99 pts/GW and still clears both tests. That refutes the earlier worry that the whole effect was the perishable rule.

## The honest caveat: the edge is decaying

The pooled +4.99 is not the edge you would get today. Split by era (2018-19..2021-22 vs 2022-23..2025-26):

| era | gain/GW | t p | Wilcoxon p | significant? |
|---|---|---|---|---|
| **2018-19..2021-22** | **+6.81** | 0.0000 | 0.0000 | ✅ |
| **2022-23..2025-26** | **+3.16** | 0.0270 | 0.0407 | ✅ |

The edge was **+6.81/GW** when few managers used xG and is **+3.16/GW** now that xG tools are mainstream — consistent with the FPL market becoming more efficient. The strong pooled significance is powered substantially by the older seasons. The realistic *forward-looking* edge is the recent end (~+3/GW), not +5.

## Every model, playing the real game

| model (points/GW, £100m squad)   |   no captain |   with captain |
|:---------------------------------|-------------:|---------------:|
| ours                             |        52.47 |          59.32 |
| player_ppg                       |        48.14 |          54.32 |
| player_form5                     |        46.41 |          51.99 |
| global_mean                      |         6.90 |           6.90 |

## What shape does each model play?

| formation (share of gameweeks)   |   ours |   player_ppg |
|:---------------------------------|-------:|-------------:|
| 3-4-3                            |   0.38 |         0.35 |
| 3-5-2                            |   0.33 |         0.32 |
| 4-3-3                            |   0.08 |         0.08 |
| 4-5-1                            |   0.06 |         0.10 |
| 5-4-1                            |   0.06 |         0.01 |
| 4-4-2                            |   0.04 |         0.06 |
| 5-3-2                            |   0.04 |         0.05 |
| 5-2-3                            |   0.02 |         0.02 |

## Verdict

**Strongest result in the project, and it is not the new rule.** The pre-specified endpoint passes (+4.99 pts/GW), 6/6 cells survive correction, AND the edge stays significant with the defensive-contribution season removed (+4.99 pts/GW over 230 gameweeks). Every model was handed the same optimizer, proven optimal against exhaustive enumeration, so *'the team-picker favoured one side'* is ruled out by construction. What remains is a difference in the **projections**. **But it is decaying** — +6.81/GW in 2018-19..2021-22 versus +3.16/GW in 2022-23..2025-26, so the realistic forward edge is the recent end, not the pooled figure.

Replication: positive in **8/8 seasons** independently, individually significant in **3/8**.

## Caveats

- The squad is rebuilt from scratch every gameweek. A real manager carries a squad and pays for transfers, so the absolute totals are unreachable. Every model faces the identical rule, so the comparison is fair.
- The captain is the highest-projected player **in the chosen XI**, not jointly optimised with it. That is the order a manager decides in.
- Autosubs (a bench player replacing a starter who played 0 minutes) are not modelled. They would help every model.
- Injury flags are unavailable historically; live they are used, which should favour our projection.
- Where projections tie, several squads are equally optimal to the model but score differently in reality; tie-breaking is arbitrary, so these p-values carry a little irreducible noise beyond sampling.