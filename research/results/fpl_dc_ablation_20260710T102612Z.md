# FPL: what is the 2025/26 edge made of? - 20260710T102612Z

Our projection beats `player_ppg` significantly in exactly one of four replayed seasons: 2025-26. That is also the only season with FPL's defensive-contribution rule, which we model and the baselines do not. This ablation forces `dc_per_90` to zero so the projection cannot see the rule. Actual points are untouched.

| run | our XI | player_ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |
|---|---|---|---|---|---|---|---|
| **with** DC (as shipped) | 53.18 | 46.76 | **+6.42** | 23/33 | 0.0082 | 0.0166 | yes |
| **without** DC (ablated) | 49.33 | 46.76 | **+2.58** | 19/33 | 0.3039 | 0.3036 | NO |

## Verdict

**H-DC.** Modelling the defensive-contribution rule is worth **+3.85 points per gameweek**, i.e. **60%** of the +6.42 edge. Without it, 2025-26 falls to +2.58 pts/GW (t p=0.3039, Wilcoxon p=0.3036) - no longer significant, and statistically indistinguishable from the three seasons without the rule.

**What this means, stated plainly:**

- The *fixture* model - clean-sheet probability, opponent strength - is worth about +2.58 pts/GW and has **never reached significance in any season**. It is not, on this evidence, a proven edge.
- The edge that *is* significant comes from correctly modelling a **new rule** before the field has adapted to it. That is genuine but **perishable**: `player_ppg` already absorbs defensive-contribution points through each player's realised history, only with a lag. Expect decay.
- It rests on a **single season** (33 gameweeks). It cannot be replicated, because 2025/26 is the only season the rule has ever existed.