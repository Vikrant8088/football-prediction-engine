# Football Prediction Feature Analysis

## Objective

The objective of this document is to identify, evaluate and prioritize every feature that could influence the outcome of a football match.

Every feature will be evaluated using three criteria:

1. Predictive Power
2. Data Availability
3. Implementation Complexity

No feature should be implemented before understanding its expected contribution to prediction accuracy.

The goal is to maximize predictive performance while minimizing unnecessary complexity.

---

## Feature Evaluation Scale

Predictive Power

★★★★★ = Extremely Important

★★★★☆ = Important

★★★☆☆ = Useful

★★☆☆☆ = Small Improvement

★☆☆☆☆ = Negligible

---

Implementation Complexity

Low

Medium

High

Very High

---

Data Availability

Free

Paid

Requires Collection

Requires Modelling

---

Every feature introduced into the prediction engine should justify its existence through measurable improvements during backtesting.

---

## Scored Feature Table

Ordered by predictive power, then by effort. "Phase" maps to
[../docs/02_FEATURE_ROADMAP.md](../docs/02_FEATURE_ROADMAP.md). A ★-rating here
is a *prior* — an expectation to be confirmed or refuted by the walk-forward
backtest, never a licence to ship the feature unproven.

| # | Feature | Predictive Power | Data Availability | Complexity | Source | Phase |
|---|---|---|---|---|---|---|
| 1 | **Team strength rating (Elo)** | ★★★★★ | Requires Modelling | Low | derived from results (+ ClubElo) | 0 ✅ |
| 2 | **Expected Goals (xG / xGA)** | ★★★★★ *(evidence: better input than goals; see note)* | Free · Understat JSON endpoint | Medium | Understat | 1 ✅ |
| 3 | **Attack / Defense strength (Poisson)** | ★★★★☆ | Requires Modelling | Low | derived from goals | 0 ✅ |
| 4 | **Home advantage** | ★★★★☆ | Requires Modelling | Low | derived | 0 ✅ |
| 5 | **xG-based form (rolling, time-decayed)** | ★★★★☆ | Requires Collection | Medium | derived from xG | 1 |
| 6 | **Key player injuries / suspensions** | ★★★★☆ *(raw count tested & rejected; see log)* | Paid (free tier: 2022+) | Medium–High | API-Football | 3b ⚠️ |
| 7 | **Confirmed pre-match lineups** | ★★★★☆ | Paid | High (timing: ~1h pre-kickoff) | API-Football | 3 |
| 8 | **Recent results form** | ★★★☆☆ | Free | Low | derived from results | 3a ✅ built |
| 9 | **Squad market value** | ★★★☆☆ | Free · Requires Collection | Medium | Transfermarkt (scrape) | 2 |
| 10 | **Rest days / fixture congestion** | ★☆☆☆☆ *(REJECTED — see evidence log)* | Free | Low | derived from fixture calendar | 3a ❌ |
| 11 | **Newly-promoted team handling (cold start)** | ★★★☆☆ | Requires Modelling | Medium | derived | 2 |
| 12 | **European / cup competition load** | ★★☆☆☆ | Paid | Medium | API-Football | 3 |
| 13 | **Match stakes / motivation (table context)** | ★★☆☆☆ | Free | Medium | derived from standings | 3 |
| 14 | **Head-to-head history** | ★★☆☆☆ | Free | Low | derived | 2 |
| 15 | **Referee tendencies** | ★★☆☆☆ | Free · Requires Collection | Medium | football-data.co.uk / FBref | later |
| 16 | **Manager change** | ★★☆☆☆ | Free · Requires Collection | Medium | Transfermarkt / news | 3 |
| 17 | **Travel distance** | ★☆☆☆☆ | Free | Low | derived from geo | later |
| 18 | **Weather** | ★☆☆☆☆ | Free | Low | weather API | later |

---

## Special case: bookmaker / market odds

| Feature | Predictive Power | Role |
|---|---|---|
| **Closing bookmaker odds** | ★★★★★ | **Benchmark only — never a model input** ✅ *measured, see below* |

The betting market is the single strongest predictor of match outcomes because
it aggregates the information of everyone betting. But the project's vision
lists "copy bookmaker predictions" as an explicit **non-goal**. The resolution:

- ❌ Never feed odds into a model — that is copying, not predicting.
- ✅ Use closing odds as the **yardstick to beat**. If the engine cannot beat
  the closing line, that is the clearest possible signal that more real
  predictive work remains.

Opening-vs-closing line movement is itself a data point (it reveals where
sharp money went), but the same non-goal applies: measure against it, don't
ingest it.

**Measured** *(Phase 4a, run `odds_report_20260710T055226Z`, 2,660 EPL matches)*:
Pinnacle closing odds score log loss 0.9464 (55.9% accuracy) vs our champion
ensemble's 0.9821 (53.1%). The market wins by 0.0357 (p<0.0001) — as expected.
The engine has captured **70% of the baseline→market edge**, and is *better
calibrated* than the market on home wins and draws; the market's advantage is
sharpness. Crucially, the closing line is priced minutes before kickoff and so
already knows confirmed line-ups and late team news, which our engine does not —
so part of the residual gap is an information advantage, not modelling skill.
This bounds what any further feature can add and points the remaining signal at
**team news**, not at weather/referees.

---

### Evidence log — Expected Goals (Phase 1, run `xg_report_20260709T103551Z`)

On 8 walk-forward EPL seasons (3040 matches), fitting team strength on **xG
beats fitting on actual goals** in *both* model families tested — Poisson
(1.0198 vs 1.0254) and Dixon-Coles (1.0107 vs 1.0241) — confirming the ★★★★★
prior that xG is a superior input signal. Adding structure compounds the gain:
`dixon_coles_xg` (xG + time-decay + low-score correction) is the **best
scoreline model of all** and the best-calibrated strength model (ECE_home
0.0264), closing roughly half of Elo's lead over Dixon-Coles. However, even the
best scoreline model does **not** beat the Elo champion (0.9956) on this window.
Conclusion: xG is a validated input, carried into the Phase 2 ensemble
alongside Elo (a strong, structurally different model) rather than crowned a
standalone champion.

### Evidence log — Injuries, importance-weighted (Phase 3c, run `injuries_report_20260710T045427Z`)

Fixed the blunt-count problem: each absence is now weighted by the player's
share of a full season's minutes **in the previous season** (leakage-safe; free,
from Understat payloads already on disk — no Transfermarkt scrape). Validation:
71.8% of absences matched a prior-season player, and the top-weighted absences
are the ever-present regulars (Raya, Lloris, Alisson, Trippier).

Controlled 3-way test on 760 matches: form 1.0268 | form+weighted 1.0361 |
form+count 1.0392. **Weighting beat the raw count**, confirming the Phase 3b
diagnosis that *who* is missing carries more than *how many* — but the gap is not
significant, and neither encoding beat form alone. Crucially, form-vs-weighted is
**also not significant** (t p=0.068, Wilcoxon p=0.249), so the honest reading is
**"no detectable injury effect in 760 matches"**, not "injuries don't matter."

The test is underpowered (2 seasons, and 3 extra features cost variance on a
small training set), and the minutes proxy scores brand-new signings at 0 even
when they are stars (van de Ven, Udogie). Injuries stay out of the champion; the
★★★★☆ prior survives as *unresolved*. To settle it: more seasons (paid tier) plus
a **market-value** importance proxy. See [[squad-market-value]].

### Evidence log — Injuries, raw count (Phase 3b, run `injuries_report_20260709T124657Z`)

Real injury data (API-Football free tier, 2022/23-2024/25) joined onto matches
(100% join accuracy). Controlled test on 760 matches: logistic model on form
(1.0268) vs form + injury-count (1.0392) — the raw count made prediction
**worse** and not significantly different. **Rejected as tested**, but with two
big asterisks that keep the ★★★★☆ prior alive: (1) only 2 seasons — underpowered;
(2) a raw count is blunt — it weights a missing star equally with a reserve, and
~3.7 flagged absences/side are mostly long-term ones already reflected in form
and ratings (the signal that matters — a *fresh* key absence — is drowned out).
The real test needs player-importance weighting (squad market value), a focus on
recent absences, and confirmed line-ups, over more seasons (paid plan). See
[[player-importance]] follow-up. Do not conclude "injuries don't matter" from
this — only "a raw count of absences, on 2 seasons, doesn't".

### Evidence log — Tiredness / rest & congestion (Phase 3a, run `features_report_20260709T114814Z`)

Controlled test on 8 walk-forward EPL seasons: an identical logistic model fit
on form features only (log loss 1.0207) vs form + rest-days + fixture-congestion
(1.0311). Adding the tiredness features made prediction **slightly worse**, and
the difference is **not significant** (Wilcoxon p=0.99). **Rejected** — its
★-rating is downgraded from ★★★☆☆ to ★☆☆☆☆ to reflect what the data showed, and
it is left out of the models. Interpretation: at EPL level, *calendar rest* is a
weak, well-managed signal; the availability of *specific players* (injuries /
line-ups) is the real edge — which is why Phase 3b (paid API data) is the
priority, not more calendar-derived features.

## How this table is used

1. Features are built in **predictive-power order, gated by effort** — Phase 1
   xG before Phase 3 lineups, even though both are highly rated, because xG is
   cheaper to acquire and validate.
2. A ★-rating is a **hypothesis**. When a feature is built, the walk-forward
   backtest either confirms it (promote, document the win) or refutes it
   (reject, document the null result — and update this table's rating to reflect
   what the data actually showed).
3. This document is **living**: every confirmed or rejected feature updates the
   relevant row, so the table always reflects evidence, not initial intuition.
