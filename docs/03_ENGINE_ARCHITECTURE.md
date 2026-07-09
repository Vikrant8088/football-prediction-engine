# Prediction Engine Architecture

> Status: **design blueprint** (no production code yet). This document is the
> plan the `prediction-engine/` subsystem will be built against. It is
> deliberately written before implementation, per the project's
> "design before coding, document before implementing" rule.

---

## 1. What the engine predicts

The engine does **not** predict a single scoreline like "2-1". It predicts a
**probability distribution over every plausible scoreline**, from which every
other prediction is derived:

```
        away goals →
        0     1     2     3    ...
home 0  4%    6%    4%    1%
 ↓   1  7%   11%    7%    2%      ──►  derived from this ONE grid:
     2  6%    9%    5%    1%           • most likely score      (argmax cell)
     3  3%    4%    2%    1%           • P(Home) / P(Draw) / P(Away)  (sum regions)
     ...                              • Over/Under 2.5 goals
                                      • Both-teams-to-score
                                      • Correct-score odds, Asian handicap, etc.
```

This is the project's founding principle made concrete: *"Exact score
prediction is a consequence of probability modelling."* Every product surface
(website, API, betting-market comparison) reads from this one grid. A model's
job is to produce the grid; everything downstream is arithmetic.

The engine also always emits, alongside the grid:
- an **explanation** (which factors moved this prediction and by how much), and
- a **calibration guarantee** (the probabilities are backtested to be honest).

---

## 2. Layered architecture

Data flows one direction, left to right. Each layer has a single
responsibility and a stable contract with the next, so any one layer can be
swapped or extended without touching the others.

```
┌───────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────────┐
│ 1. INGESTION  │──►│ 2. FEATURE    │──►│ 3. MODELS    │──►│ 4. ENSEMBLE  │──►│ 5. SERVING    │
│    raw lake   │   │    STORE      │   │  (scoreline  │   │  + CALIBRATE │   │  API / report │
│               │   │  point-in-    │   │   grids)     │   │              │   │               │
│               │   │  time correct │   │              │   │              │   │               │
└───────────────┘   └───────────────┘   └──────────────┘   └──────────────┘   └───────────────┘
        │                                        ▲                  ▲
        │                                        │                  │
        └────────────────────────────────────────┴──────────────────┘
                          6. EVALUATION / BACKTEST GATE
             (walk-forward, proper scoring rules, significance —
              nothing reaches serving unless it beats the champion here)
```

### Layer 1 — Ingestion (`data_warehouse/`) — *partially built*
Acquires raw data from the internet into a **versioned, immutable raw lake**
with recorded provenance (source URL, timestamp, checksum). Already exists for
football-data.co.uk; will grow one module per new source (xG, injuries,
lineups, squad values). One rule: **never mutate or delete a snapshot**, so any
past backtest is exactly reproducible.

### Layer 2 — Feature Store (`prediction-engine/features/`) — *new*
Turns raw snapshots into **model-ready features**, and this is where most of
the "intelligence" lives. The non-negotiable constraint:

> **Point-in-time correctness.** Every feature for a match kicking off at time
> *T* must be computable using *only* information available before *T*. No
> feature may leak a result, a final xG, or a lineup that was not yet public.

Feature builders are pure functions `(match, history_before_T) -> value`, unit-
tested for leakage. The store materializes a feature table keyed by
`(match_id, feature_name, as_of_time)` so both training and live prediction
read the *same* code path — eliminating train/serve skew.

### Layer 3 — Models (`prediction-engine/models/`, candidates in `research/experiments/`) — *partially built*
Every model implements one contract (already defined in
[research/experiments/base.py](../research/experiments/base.py)): `fit(history)`
then `predict_proba(fixtures)`. We extend the contract with `predict_scoreline_grid()`
so score-level outputs are first-class. Model families:

| Family | Examples | Strength |
|---|---|---|
| **Rating** | Elo (current champion), Glicko | simple, robust, great baseline |
| **Statistical goals** | Poisson, Dixon-Coles, **bivariate Poisson** | native scoreline grids, explainable |
| **Machine learning** | LightGBM / XGBoost on the feature store | absorbs many weak signals at once |
| **Bayesian** (later) | hierarchical Poisson (PyMC) | uncertainty + shrinkage, very explainable |

ML models predict goal *rates* (λ_home, λ_away) or a scoreline grid — never a
naked class label — so the scoreline-distribution invariant always holds.

### Layer 4 — Ensemble + Calibration (`prediction-engine/ensemble/`) — *new*
No single model wins everywhere. This layer **blends** model grids (stacking /
weighted average, weights themselves fit by walk-forward CV) and then
**calibrates** the blended output so stated probabilities match observed
frequencies (isotonic / Platt scaling, validated by the existing
[calibration harness](../research/evaluation/calibration.py)). Output: the final,
honest scoreline grid.

### Layer 5 — Serving (`prediction-engine/serving/`) — *new, last*
A thin API (FastAPI) that, given a fixture, returns the scoreline grid, the
derived markets, and the explanation. **Built last, on purpose** — per the
vision, "prediction quality takes priority over UI and monetisation."

### Layer 6 — Evaluation / Backtest Gate (`research/evaluation/`) — *built*
The governance layer, already implemented and rigorous: walk-forward expanding-
window backtesting, proper scoring rules (log loss / RPS / Brier), calibration
error, and paired significance tests. **This is the gate:** a new model,
feature, or ensemble ships *only if* it beats the reigning champion here, with
the margin reported honestly (including when it is *not* significant).

---

## 3. The governing loop (why this is "smart", not just "big")

Intelligence is not any single fancy model — it is the **closed improvement
loop**:

```
   hypothesis ("adding xG-form will improve accuracy")
        │
        ▼
   build feature (point-in-time correct)  ──►  add to a candidate model
        │
        ▼
   walk-forward backtest vs current champion
        │
    ┌───┴────────────────┐
    │ beats champion?    │
    ├── yes ─► promote: new champion, document the win, ship
    └── no  ─► reject: document the null result, keep the feature out
```

A feature that does not measurably help is *rejected and recorded as rejected*.
This is what keeps the engine improving instead of just accreting complexity.

---

## 4. Proposed repository layout

```
data_warehouse/            # Layer 1 — ingestion (exists)
  sources/
    football_data_co_uk.py #   (exists)
    understat.py           #   + xG event/shot data          [Phase 1]
    fbref.py               #   + xG, advanced team stats      [Phase 1]
    api_football.py        #   + injuries, lineups, fixtures  [Phase 3]
    transfermarkt.py       #   + squad market values          [Phase 2]

prediction-engine/         # Layers 2–5 — the production engine (empty today)
  features/                #   feature store + builders       [Phase 2]
  models/                  #   promoted, production models
  ensemble/                #   blending + calibration         [Phase 2]
  serving/                 #   FastAPI prediction service     [Phase 4]

research/                  # the lab (exists)
  experiments/             #   candidate models (baseline/elo/poisson/dixon_coles)
  evaluation/              #   Layer 6 — backtest gate (exists)
  data/                    #   research-grade loaders
  results/                 #   timestamped, reproducible reports
```

`research/` is the **lab**: where candidate models and features are proven.
`prediction-engine/` is the **factory**: only proven components graduate into
it. This separation is deliberate — experiments stay messy and fast; production
stays clean and gated.

---

## 5. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Core numerics | numpy, scipy, pandas | already in use; explicit, inspectable |
| Statistical fits | scipy.optimize (MLE) | no black box — every parameter is named (project principle) |
| ML models | LightGBM / XGBoost | strong tabular performance, feature importances = explainability |
| Bayesian (later) | PyMC | uncertainty + hierarchical shrinkage |
| Scraping | httpx + selectolax/BeautifulSoup; Playwright only if JS-rendered | Understat/FBref |
| Real-time data | API-Football client | injuries, confirmed lineups, live fixtures |
| Serving | FastAPI | last, thin |
| Experiment tracking | timestamped artifacts in `research/results/` (exists); consider MLflow later | reproducibility first |

---

## 6. Cross-cutting rules (inherited from CLAUDE.md)

1. **No leakage, ever.** Point-in-time correctness is enforced in the feature
   store and checked by tests. A leaking feature invalidates every backtest.
2. **Nothing hardcoded.** Sources, leagues, seasons, model hyperparameters,
   API keys → config, never source code.
3. **Beat the champion or don't ship.** Every change is gated by Layer 6.
4. **Explainable before black-box.** Prefer models whose outputs decompose into
   named contributions; even ML models must expose feature attributions.
5. **Reproducible artifacts.** Every experiment writes a timestamped report;
   results are never printed-and-discarded.
6. **The market is a benchmark, not an input.** Bookmaker odds are used to
   *measure* the engine, never fed into it (copying bookmakers is a non-goal).

---

## 7. Open decisions (to resolve as we build)

- **League scope:** deepen one league (E0, where the champion is established)
  before going broad, or build multi-league from the start? *(Recommend: deepen
  first — features like xG are validated faster on one league.)*
- **API-Football tier:** which plan (rate limits / historical depth) matches the
  seasons we backtest on?
- **Live vs pre-match:** confirmed-lineup features only exist ~1 hour pre-match.
  Do we support a "provisional" prediction that upgrades when lineups drop?
- **Retraining cadence:** nightly full refit, or incremental? Decide once the
  ensemble exists and we can measure drift.

See [02_FEATURE_ROADMAP.md](02_FEATURE_ROADMAP.md) for the phased build order and
[../research/FEATURE_ANALYSIS.md](../research/FEATURE_ANALYSIS.md) for the scored
factor list this architecture is designed to absorb.
