# FPL opening run: weight the fixture run, or pick on GW1? — 20260801T105534Z

Hold a squad for N gameweeks, no transfers, fielding the best XI each week and scoring on actual points. `single` picks it on the first gameweek's projection; `run` picks it on a decay-weighted projection over the whole hold. Everything the `run` squad sees is **as of the start gameweek** — only the fixture schedule is read ahead, never how those matches turned out.

## Primary (pre-specified): `run` (decay 0.85, N=5) vs `single`

> **+2.25 points per window** (254.0 vs 251.8 over 5-GW holds), **+0.45/GW**, winning 11/24 windows.
>
> paired t p=0.7477 · Wilcoxon p=0.9636 → **FAILS** the two-test rule.

Windows are non-overlapping, so this is only 24 independent holds across the seasons — low power. Read the effect size first.

## Sensitivity (Holm-corrected)

| decay | N | gain/window | gain/GW | won | t p | Wilcoxon p | both? | corrected |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 5 | **-2.62** | -0.53 | 10/24 | 0.6697 | 0.6157 | ❌ | ❌ |
| 0.70 | 5 | **+6.58** | +1.32 | 13/24 | 0.3030 | 0.3614 | ❌ | ❌ |
| 0.85 | 3 | **+7.95** | +2.65 | 24/43 | 0.0178 | 0.0341 | ✅ | ❌ |
| 0.85 | 8 | **+13.69** | +1.71 | 10/16 | 0.1062 | 0.0833 | ❌ | ❌ |

## Verdict

**Null: picking on GW1 is enough.** Over 24 independent holds the opening-run weighting gains only +0.45/GW (t p=0.7477, W p=0.9636) — not enough to clear the bar. On this evidence the single-gameweek squad is fine for the opener; the fixture run does not need special weighting, likely because strong teams tend to be strong across the run anyway and the single-GW pick already captures most of it.

## Caveats
- Non-overlapping windows -> small sample -> low power; effect size leads.
- Held with NO transfers, to isolate the opening-squad choice. A real manager transfers, which would blunt any opening-run advantage further.
- Both squads field the best XI each week by the as-of-G projection, so the comparison is purely the 15 players chosen.