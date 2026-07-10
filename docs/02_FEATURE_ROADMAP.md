# Feature Roadmap

> The phased build order for the prediction engine. Each phase has a **goal**,
> concrete **deliverables**, and a **success criterion** — and per the
> project's rules, no phase is "done" until its work **beats the current
> champion** on the walk-forward backtest (or is honestly recorded as a null
> result). Phases are ordered by *predictive value per unit of effort*, not by
> what is most exciting to build.
>
> See [03_ENGINE_ARCHITECTURE.md](03_ENGINE_ARCHITECTURE.md) for the system
> design and [../research/FEATURE_ANALYSIS.md](../research/FEATURE_ANALYSIS.md)
> for the scored factor list these phases draw from.

---

## Phase 0 — Foundation ✅ *(done)*

**Goal:** a reproducible pipeline and an established baseline champion.

- ✅ Versioned raw data lake (football-data.co.uk, 17 EPL seasons).
- ✅ Walk-forward backtest harness with proper scoring rules, calibration, and
  significance testing.
- ✅ Four benchmarked models (baseline, Poisson, Dixon-Coles, Elo).
- ✅ **Champion: Elo** (log loss 0.9928), significant over the field.

**Current champion to beat:** Elo, log loss **0.9928**, RPS **0.2065**.

---

## Phase 1 — Expected Goals (xG) ✅ *(done — xG signal validated; Elo still champion)*

**Goal:** replace noisy actual-goals signal with chance-quality signal.

**Why first:** xG is the highest-predictive-power feature available (see feature
analysis). Actual goals are noisy; xG measures the quality of chances a team
creates and concedes, and predicts future results markedly better. This is the
largest accuracy jump on the board.

**Deliverables — built**
- ✅ `data_warehouse/sources/understat.py` — ingests per-match xG/xGA (and
  goals) from Understat's JSON endpoint into the versioned lake (12 EPL
  seasons, 2014/15–2025/26).
- ✅ `research/data/xg_loader.py` — self-contained match+xG dataset (no
  cross-source join needed; Understat carries goals too).
- ✅ `research/experiments/poisson_xg.py` — the **xG-aware scoreline model**:
  fits attack/defense/home-advantage on xG via Poisson deviance, still emits
  the full actual-goal scoreline grid.
- ✅ `research/evaluation/benchmark_xg.py` — walk-forward benchmark, all models
  scored on the identical Understat match set for a controlled contrast.
- ✅ `research/experiments/dixon_coles_xg.py` — the follow-up: xG strengths +
  Dixon-Coles's two innovations (time decay + low-score correction), fitted in
  two stages (strengths from recency-weighted xG, rho from actual goals).

**Result** *(run `xg_report_20260709T110624Z`, 8 eval seasons, 3040 matches)*

| model | log loss | ECE(home) | note |
|---|---|---|---|
| **elo** (goals) | **0.9956** | 0.0373 | overall champion (unchanged from Phase 0) |
| dixon_coles_xg (**xG**) | 1.0107 | **0.0264** | best scoreline model; best-calibrated of the strength models |
| poisson_xg (**xG**) | 1.0198 | 0.0322 | |
| dixon_coles (goals) | 1.0241 | 0.0423 | |
| poisson (goals) | 1.0254 | 0.0439 | |

Two effects, both clean and consistent:
- ✅ **xG beats goals** in *both* families (poisson_xg < poisson; dixon_coles_xg
  < dixon_coles) — the input-signal result replicates.
- ✅ **Structure matters too:** dixon_coles_xg (xG + time-decay + tau) is the
  best scoreline model of all, and **closes ~half the gap to Elo** (Elo's lead
  over Dixon-Coles shrank from −0.0285 to −0.0151 in per-match log loss once xG
  and time-decay were added). It is also the **best-calibrated** strength model.
- ❌ **Still short of Elo (0.9956).** Even the strongest scoreline structure +
  the best input doesn't overtake Elo's rating dynamics on this window.
- **Honest conclusion:** we now have two strong, *different* model families —
  Elo (rating dynamics) and Dixon-Coles-xG (scoreline distribution). Neither
  strictly dominates the other across all metrics. That is exactly the setup an
  **ensemble** exploits → Phase 2.

**Champion after Phase 1:** still Elo (0.9956). Best scoreline model:
dixon_coles_xg (1.0107). Both carried into Phase 2 as ensemble components.

---

## Phase 2 — Feature store + ML model + ensemble 🔄 *(in progress — ensemble done, new champion)*

**Goal:** absorb many weak signals at once, and blend models.

### Phase 2a — Ensemble ✅ *(done — new champion)*

- ✅ `research/experiments/ensemble.py` — weighted blend of the strong, diverse
  base models (Elo + Poisson-xG + Dixon-Coles-xG). Weights minimise log loss
  via a softmax parameterisation, fit **only on an inner temporal split of each
  fold's training data** (no leakage); weights are inspectable and logged.
- ✅ `research/evaluation/benchmark_ensemble.py` — Phase 2 benchmark + report.

**Result** *(run `ensemble_report_20260709T112538Z`, 8 seasons, 3040 matches)*

| model | log loss | RPS | Brier | ECE(draw) |
|---|---|---|---|---|
| **ensemble** 👑 | **0.9925** | **0.2077** | **0.5912** | **0.0022** |
| elo (prev champion) | 0.9956 | 0.2083 | 0.5926 | 0.0072 |
| dixon_coles_xg | 1.0107 | 0.2130 | 0.6031 | 0.0071 |
| poisson_xg | 1.0198 | 0.2171 | 0.6101 | 0.0093 |

- ✅ **Ensemble is the new champion** — best on *every* proper scoring rule
  (log loss, RPS, Brier) **and** better calibrated than Elo on all three
  outcomes. The success criterion is met.
- ✅ **It genuinely blends:** average learned weights across folds are Elo 72%,
  Poisson-xG 12%, Dixon-Coles-xG 16% — the xG models get real weight in 7 of 8
  seasons, confirming the Phase 1 setup (two strong, complementary views).
- ⚠️ **But the margin is small and not robustly significant.** The edge over
  Elo (0.9925 vs 0.9956) is significant on Wilcoxon (p=0.0002) but **not** on
  the paired t-test (p=0.06). Treat the ensemble as the new *provisional*
  champion, not a decisive leap.
- ⚠️ **Diminishing returns signal:** in the most recent fold the blend
  collapsed to ~100% Elo. Reblending the *existing* models is near its
  ceiling — the next real gains need genuinely new **information**, not new
  combinations. That is the case for Phase 3.

### Phase 2b — Feature store + ML model *(pending)*

- `prediction-engine/features/` — point-in-time-correct feature builders
  (ratings, xG-form, rest days, congestion, promotion cold-start, head-to-head),
  with leakage tests.
- `data_warehouse/sources/transfermarkt.py` — squad market values.
- An **ML model** (LightGBM) predicting goal rates from the feature store, added
  as a candidate and folded into the ensemble.

**Champion after Phase 2a:** **ensemble** (0.9925) — provisional, pending a
more significant margin or new signals.

---

## Phase 3 — Real-time context (injuries, lineups, stakes) 🔄 *(in progress)*

**Goal:** the signals only live/paid data provides — the "smart" edge.

### Phase 3a — Feature store + tiredness signal ✅ *(done — tiredness rejected)*

Built the point-in-time-correct **feature store** (the machinery every Phase 3
signal plugs into) and used it to test the *free* half of Phase 3: tiredness.

- ✅ `research/features/builders.py` — leakage-tested feature builders (form,
  **rest days**, **fixture congestion**). The no-leakage test asserts a match's
  features never change when later matches are added.
- ✅ `research/experiments/feature_logistic.py` — an explainable multinomial
  logistic model (MLE, not a black box) that consumes arbitrary features — the
  vehicle for testing any new signal.
- ✅ `research/evaluation/benchmark_features.py` — controlled experiment.

**Result** *(run `features_report_20260709T114814Z`, 8 seasons, 3040 matches)*

| model | log loss | note |
|---|---|---|
| ensemble (champion) | 0.9925 | (context) |
| elo | 0.9956 | (context) |
| logistic_form | 1.0207 | form features only |
| logistic_form_rest | 1.0311 | form + **tiredness** — *worse* |

- ❌ **Tiredness (rest + congestion) REJECTED.** Adding it to form features did
  **not** improve prediction — it was slightly *worse* (1.0311 vs 1.0207), and
  the difference is not significant (Wilcoxon p=0.99). Per project discipline,
  it is documented as rejected and left out. (This matches football-analytics
  findings: calendar rest is a weak signal — top squads rotate around it.)
- 💡 **What this tells us:** *who is on the calendar* doesn't matter much; *who
  is on the pitch* does. Generic tiredness isn't the edge — **injuries and
  confirmed line-ups are.** That is the paid half of Phase 3, and this negative
  result sharpens the case for it.

### Phase 3b — Injuries (raw count) 🔄 *(tested on free data — count rejected; needs a smarter feature)*

Wired up real injury data (free API-Football tier) and tested the simplest
possible availability feature: how many players each side is missing.

- ✅ `data_warehouse/sources/api_football.py` — authenticated, paginated source
  (secret key via `APIFOOTBALL_KEY` env var, never committed); injuries stored
  in the versioned lake. `BaseDataSource` gained a `_fetch_content` hook so
  paginated APIs reuse all the versioning/skip logic.
- ✅ `research/data/injury_loader.py` — joins injuries onto matches (100% of
  2,184 injury keys matched a real match; 3-team name map), producing
  home/away missing-player counts, leakage-safe (NaN outside covered seasons).

**Result** *(run `injuries_report_20260709T124657Z`, 2 seasons, 760 matches)*

| model | log loss | note |
|---|---|---|
| elo | 0.9789 | (context, best on this small window) |
| ensemble | 0.9824 | (context) |
| logistic_form | 1.0268 | form features only |
| logistic_form_injuries | 1.0392 | form + **injury count** — *worse* |

- ❌ **Raw injury COUNT rejected.** Adding "number of players missing" to form
  did not improve prediction (1.0392 vs 1.0268; not significant). Same shape as
  the tiredness result.
- ⚠️ **But this is a deliberately blunt test, not a verdict on injuries.** Two
  real limitations: (1) only **2 seasons / 760 matches** (all the free plan
  allows) — underpowered; (2) a raw count treats a missing **star striker** the
  same as a missing **3rd-choice full-back**, and ~3.7 players/side are flagged
  per match — most are long-term absentees **already absorbed into form and
  ratings**. The count is dominated by noise; the real signal (a key player
  *freshly* out) is buried in it.
- 💡 **What would actually test injuries properly:** weight absences by **player
  importance** (minutes played / market value), focus on *recent* absences, and
  add **confirmed line-ups** (who actually starts). That needs player-value data
  and more seasons (paid plan). Until then, the blunt count stays out.

**Champion after Phase 3b:** unchanged — **ensemble** (0.9925 on the full
window). No new signal has beaten it yet.

### Phase 3c — Importance-weighted absences ✅ *(done — diagnosis confirmed, signal still not additive)*

Phase 3b's diagnosis was that a raw count is blunt. Phase 3c fixed the encoding
and re-ran the identical test, so the *only* thing that changed was how the
absence is weighted.

- ✅ `research/data/player_importance.py` — importance = the player's share of a
  full season's minutes **in the PREVIOUS season** (leakage-safe by
  construction). Sourced free from the Understat payloads already in the lake —
  **no Transfermarkt scrape and no paid data needed**. Name matching reduces
  "D. de Gea" and "David de Gea" to one key; 71.8% of absences matched, and the
  top-weighted absences are exactly the ever-present regulars (Raya, Lloris,
  Alisson, Trippier), while youth/new signings score 0.

**Result** *(run `injuries_report_20260710T045427Z`, 2 seasons, 760 matches)*

| model | log loss | note |
|---|---|---|
| elo | 0.9789 | (context) |
| ensemble | 0.9824 | (context) |
| **logistic_form** | **1.0268** | the bar to beat |
| logistic_form_injury_weight | 1.0361 | + **importance-weighted** absences |
| logistic_form_injury_count | 1.0392 | + raw count (Phase 3b) |

- ✅ **The diagnosis was right, directionally.** Weighting by importance beats
  the raw count (1.0361 vs 1.0392) — encoding the *who*, not just the *how many*,
  does carry more information. But the gap is **not significant** (p=0.17–0.56).
- ❌ **Still not additive.** Neither encoding beats form alone (1.0268). And
  critically, those gaps are **not statistically significant either**
  (form-vs-weighted: t p=0.068, Wilcoxon p=0.249) — so the honest statement is
  not "injuries hurt" but **"we cannot detect an injury effect in 760 matches."**
- ⚠️ **This test is underpowered, and we know it.** 760 evaluation matches, and
  adding 3 features to a model trained on ~380–760 rows costs variance. A real
  effect of plausible size would be invisible here.
- ⚠️ **Known flaw in the importance proxy:** previous-season minutes give a
  brand-new signing 0 importance even if he is a star (van de Ven, Udogie both
  scored 0). Squad **market value** would capture those; minutes cannot.

**Verdict:** injuries stay **out of the champion** on current evidence — but the
result is *inconclusive*, not a refutation. Resolving it needs (a) more seasons
(paid API tier, ~6), and (b) a market-value importance proxy that handles new
signings. Both are cheap-ish; neither is justified until something else stalls.

**Champion after Phase 3:** unchanged — **ensemble** (0.9925).

---

## Cross-cutting — Multi-league generalization ✅ *(done — champion confirmed, and now significant)*

Everything up to this point was established on the Premier League alone. A
champion that only wins in England is an overfitted champion, so the same
walk-forward benchmark was re-run **independently on each of Europe's top five
leagues** (each trained and evaluated strictly within itself — no cross-league
training). Pooling the five at *scoring* time also gives ~5x the matches, which
finally supplies the statistical power the single-league test lacked.

**Result** *(run `multileague_report_20260710T052635Z`, 5 leagues, 14,284 matches/model)*

| model | EPL | La Liga | Bundesliga | Serie A | Ligue 1 | **pooled** |
|---|---|---|---|---|---|---|
| **ensemble** 👑 | **0.9925** | **0.9945** | 0.9988 | **0.9963** | **1.0212** | **1.0003** |
| elo | 0.9956 | 0.9995 | 1.0064 | 0.9971 | 1.0333 | 1.0058 |
| poisson_xg | 1.0198 | 0.9968 | **0.9985** | 1.0099 | 1.0224 | 1.0096 |
| dixon_coles_xg | 1.0107 | 1.0023 | 1.0089 | 1.0240 | 1.0267 | 1.0145 |
| baseline | 1.0673 | 1.0707 | 1.0736 | 1.0842 | 1.0763 | 1.0744 |

- ✅ **The champion generalizes.** The ensemble wins **4 of 5** leagues outright,
  and in the fifth (Bundesliga) it loses to poisson_xg by 0.0003 — a hair. It is
  best-or-tied-best everywhere. It was not an artefact of the Premier League.
- ✅ **The champion is now statistically significant.** Pooled over 14,284
  matches, ensemble (1.0003) beats Elo (1.0058) on **both** tests
  (paired t p<0.0001, Wilcoxon p<0.0001). The margin that was only borderline on
  the EPL (t p=0.06) holds up decisively at 5x the sample. **The ensemble is no
  longer "provisional" — it is the settled champion.**
- 📊 **League texture worth knowing:** Elo is clearly weakest in Ligue 1 (1.0333,
  behind both xG models), while the xG models are relatively strongest in
  Bundesliga and La Liga. The ensemble's value is precisely that it re-weights
  toward whichever view works in a given league.

**Champion:** **ensemble** — confirmed across 5 leagues, significant, and better
calibrated than any component.

---

## Cross-cutting — Hyperparameter tuning ✅ *(done — defaults were already near-optimal)*

The outstanding milestone flagged in the very first Phase 0 report: every model
had used **literature-default** hyperparameters (Elo K=20 / home advantage=100;
Dixon-Coles xi=0.0065/day), never fitted to this data. Now settled.

- ✅ `research/experiments/tuning.py` — `TunedModel`, a **nested walk-forward**
  search. On each fold, candidate settings are scored on an inner validation
  split of the *training* window only, so a setting is never chosen by looking
  at the season it predicts. Drops into the harness (and into the ensemble as a
  base) like any other model, and reports the settings it picked.

**Result** *(run `tuning_report_20260710T051002Z`, 8 seasons, 3040 matches)*

| model | default | tuned | verdict |
|---|---|---|---|
| elo | **0.9956** | 0.9963 | tuning does not help |
| dixon_coles_xg | **1.0107** | 1.0151 | tuning does not help |
| **ensemble** | **0.9925** 👑 | 0.9943 | tuning does not help |

- ❌ **Tuning made every model marginally worse**, and no difference was
  significant (paired t p=0.12–0.57). Settings the search landed on (Elo K=25,
  home advantage=120; xi=0.002) beat the defaults on the inner validation split
  but did **not** transfer to the next season — textbook mild overfitting of the
  hyperparameter search itself.
- ✅ **This is a useful null result, not a wasted pass.** It closes the Phase 0
  milestone, confirms the published defaults are already near-optimal for this
  data, and **rules tuning out as a lever** so we stop wondering about it.

**Champion after tuning:** unchanged — **ensemble** (0.9925) with default
hyperparameters.

---

## Phase 4 — Serving + continuous improvement

**Goal:** expose predictions and keep the engine improving on its own.

**Deliverables**
- `prediction-engine/serving/` — FastAPI service returning the scoreline grid,
  derived markets (1X2, over/under, BTTS, correct score), and the explanation.
- Automated retraining + a **drift monitor** that re-runs the backtest on new
  results and alerts when the champion degrades.
- A standing **benchmark vs closing bookmaker odds** — the market is the true
  yardstick (used to measure, never as a model input).

**Success criterion:** live predictions stay calibrated over a full season, and
the engine measurably closes the gap to (or beats) the closing-odds benchmark.

---

## Guardrails that apply to every phase

- **Point-in-time correctness** — no feature may use information unavailable
  before kickoff. Leakage invalidates the whole backtest.
- **Beat the champion or don't ship** — every model/feature/ensemble is gated
  by the walk-forward backtest.
- **Explainability** — every prediction decomposes into named factor
  contributions.
- **Reproducibility** — every experiment writes a timestamped, re-runnable
  report to `research/results/`.
