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

**Result** *(run `xg_report_20260709T103551Z`, 8 eval seasons, 3040 matches)*

| model | log loss | vs |
|---|---|---|
| **elo** (goals) | **0.9956** | overall champion (unchanged from Phase 0) |
| poisson_xg (**xG**) | 1.0198 | best of the goals-family; beats… |
| dixon_coles (goals) | 1.0241 | |
| poisson (goals) | 1.0254 | its own goal-based twin |

- ✅ **Hypothesis supported at the signal level:** within the same Poisson
  machinery, fitting strength on **xG beats fitting on goals** (1.0198 vs
  1.0254; significant on the paired t-test, p=0.006). poisson_xg also
  calibrates better (lower ECE) than any goal model, and even edges out
  Dixon-Coles.
- ❌ **Success criterion not fully met:** the pure xG-Poisson does *not* beat
  the Elo champion (0.9956). Elo's rating dynamics remain a stronger structure
  than one-shot strength estimation, whatever the input signal.
- **Honest conclusion:** xG is a better *input*, but a better input inside a
  weaker *model structure* isn't enough. The win comes from combining xG with
  Elo-style dynamics and ensembling → Phase 2. An xG-aware Dixon-Coles
  (xG strengths + time decay + low-score correction) is the immediate follow-up.

**Champion after Phase 1:** still Elo (0.9956 on this window). The xG *signal*
is promoted into the candidate toolbox for Phase 2.

---

## Phase 2 — Feature store + ML model + ensemble

**Goal:** absorb many weak signals at once, and blend models.

**Deliverables**
- `prediction-engine/features/` — the **feature store**: point-in-time-correct
  feature builders (ratings, xG-form, rest days, congestion, promotion cold-
  start, head-to-head), with leakage tests.
- `data_warehouse/sources/transfermarkt.py` — squad market values.
- An **ML model** (LightGBM) predicting goal rates / scoreline grid from the
  feature store, added as a candidate in `research/experiments/`.
- `prediction-engine/ensemble/` — blend the statistical, rating, and ML grids
  (weights fit by walk-forward CV) + final **calibration** pass.

**Success criterion:** the calibrated ensemble beats the best single model from
Phase 1, and is better-calibrated (lower ECE) than any component alone.

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
