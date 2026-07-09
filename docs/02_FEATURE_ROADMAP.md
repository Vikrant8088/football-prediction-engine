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

## Phase 3 — Real-time context (injuries, lineups, stakes)

**Goal:** the signals only live/paid data provides — the "smart" edge.

**Deliverables**
- `data_warehouse/sources/api_football.py` — fixtures, **injuries/suspensions**,
  and **confirmed pre-match lineups** (available ~1h before kickoff).
- Availability features: key-player-out, squad-value-missing, lineup strength.
- Context features: rest asymmetry, European/cup congestion, match stakes
  (title/relegation/mid-table), manager change.
- A "provisional → upgraded" prediction flow: predict early, refine when
  lineups drop.

**Success criterion:** each context feature is added *only if* it beats the
Phase 2 ensemble; features that don't are documented as rejected and left out.

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
