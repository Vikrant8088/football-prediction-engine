# FPL: the real game — a £100m squad, not a wish list — 20260710T160739Z

Walk-forward across **4 seasons, 131 gameweeks, 96,303 player-gameweeks**. Each gameweek every model buys a legal **15-man squad** (2 GKP, 5 DEF, 5 MID, 3 FWD; max 3 per club across all fifteen; £100m), starts its best legal eleven, and scores only those eleven.

The squad is chosen by branch and bound, **verified against exhaustive enumeration** of every legal XI and every legal bench behind it. Every model gets the same optimizer, so a difference here is a difference in the **projections**, not in how the team was assembled.

> **Why not an "XI budget"?** There isn't one. The bench costs real money and its shape is forced by the formation (the squad quota is fixed), so the money left for the eleven is £83.1m–£84.3m depending on what you play. An earlier version of this benchmark swept flat XI budgets up to £90m — squads that cannot be bought — and £90m gave the best result in the sweep. Deleted.

## Primary endpoint (pre-specified)

The game as played: **£100m squad, with a captain.** Named before any of these numbers existed. A single pre-specified endpoint needs no multiplicity correction — and it is not the flattering choice, because captaincy *reduces* our measured gain.

> **+3.22 points per gameweek** (59.34 vs 56.12), winning 74/131 gameweeks, +122 across a 38-gameweek season.
>
> paired t p=0.0184 · Wilcoxon p=0.0302 → **PASSES** the two-test rule.

## Sensitivity: a poorer squad

A richer squad is not a variation — £100m is the rule. A poorer one is.

| squad budget | captain | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p | both? | **corrected** |
|---|---|---|---|---|---|---|---|---|---|
| £100m | no | 52.93 | 49.34 | **+3.60** | 75/131 | 0.0041 | 0.0145 | ✅ | **✅** |
| £100m | yes | 59.34 | 56.12 | **+3.22** | 74/131 | 0.0184 | 0.0302 | ✅ | **✅** |
| £95m | no | 51.30 | 47.04 | **+4.26** | 78/131 | 0.0011 | 0.0013 | ✅ | **✅** |
| £95m | yes | 57.65 | 53.85 | **+3.80** | 76/131 | 0.0095 | 0.0067 | ✅ | **✅** |
| £90m | no | 49.73 | 44.15 | **+5.57** | 82/131 | 0.0001 | 0.0001 | ✅ | **✅** |
| £90m | yes | 56.44 | 50.69 | **+5.76** | 81/131 | 0.0004 | 0.0006 | ✅ | **✅** |

**Multiplicity.** 6 configurations were tested, so reporting whichever clears p<0.05 would be cherry-picking. `corrected` applies **Holm-Bonferroni** across all 6, to both tests: **6/6 survive.** Written before these numbers existed, and conservative here because the cells are heavily correlated.

## Replication: season by season (primary configuration)

One pooled p-value carried by a single lucky season is what invalidated the earlier claim. These are independent replications.

| season | our XI | ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |
|---|---|---|---|---|---|---|---|
| 2022-23 | 58.78 | 59.16 | **-0.38** | 14/32 | 0.9078 | 0.7464 | ❌ |
| 2023-24 | 56.09 | 54.55 | **+1.55** | 17/33 | 0.5103 | 0.5518 | ❌ |
| 2024-25 | 63.09 | 57.94 | **+5.15** | 23/33 | 0.0701 | 0.0830 | ❌ |
| 2025-26 | 59.39 | 52.94 | **+6.45** | 20/33 | 0.0115 | 0.0258 | ✅ |

## The load-bearing check: does the edge survive without the new rule?

2025/26 is the only season with the **defensive-contribution rule**, which we model and the baselines do not. In the projection-only backtest that rule was ~60% of a *perishable* edge. So the test that matters: drop 2025/26 entirely and re-measure on the three seasons that lack the rule.

> **Three non-DC seasons (98 gameweeks), £100m + captain:** +2.13 pts/GW, winning 54/98, t p=0.1865, Wilcoxon p=0.2417 → **NOT significant.**

So of the +3.22 pooled gain, roughly 2.1 is a fixture edge present across seasons (positive but underpowered), and the rest is the one-season DC-rule advantage we already know is perishable. This is more than Phase 5b had — the non-DC signal is genuinely positive, not zero — but it does not clear the bar alone.

## Every model, playing the real game

| model (points/GW, £100m squad)   |   no captain |   with captain |
|:---------------------------------|-------------:|---------------:|
| ours                             |        52.93 |          59.34 |
| player_ppg                       |        49.34 |          56.12 |
| player_form5                     |        46.69 |          52.23 |
| global_mean                      |         5.84 |           5.84 |

## What shape does each model play?

| formation (share of gameweeks)   |   ours |   player_ppg |
|:---------------------------------|-------:|-------------:|
| 3-5-2                            |   0.41 |         0.52 |
| 3-4-3                            |   0.22 |         0.34 |
| 4-3-3                            |   0.08 |         0.01 |
| 5-3-2                            |   0.08 |         0.06 |
| 5-4-1                            |   0.07 |         0.00 |
| 4-4-2                            |   0.05 |         0.04 |
| 4-5-1                            |   0.05 |         0.02 |
| 5-2-3                            |   0.04 |         0.01 |

## Verdict

**Promising, and the strongest result so far — but not proven.** The pre-specified endpoint passes (+3.22 pts/GW) and all 6/6 cells survive correction. But drop the one defensive-contribution season and the edge falls to +2.13 pts/GW, **no longer significant** (t p=0.1865). So the pooled result leans on the same perishable rule-modelling advantage that Phase 5c already identified. Unlike Phase 5b the residual fixture edge is genuinely positive across seasons — it is just too small to prove on 98 gameweeks. Every model was handed the same optimizer, proven optimal against exhaustive enumeration, so *'the team-picker favoured one side'* is ruled out by construction. What remains is a difference in the **projections**.

Replication: positive in **3/4 seasons** independently, individually significant in **1/4**.

## Caveats

- The squad is rebuilt from scratch every gameweek. A real manager carries a squad and pays for transfers, so the absolute totals are unreachable. Every model faces the identical rule, so the comparison is fair.
- The captain is the highest-projected player **in the chosen XI**, not jointly optimised with it. That is the order a manager decides in.
- Autosubs (a bench player replacing a starter who played 0 minutes) are not modelled. They would help every model.
- Injury flags are unavailable historically; live they are used, which should favour our projection.
- Where projections tie, several squads are equally optimal to the model but score differently in reality; tie-breaking is arbitrary, so these p-values carry a little irreducible noise beyond sampling.