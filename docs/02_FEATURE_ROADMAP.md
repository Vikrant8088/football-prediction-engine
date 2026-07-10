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

### Phase 4a — Benchmark vs the closing bookmaker line ✅ *(done — 70% of the achievable edge captured)*

The true finish line, and the one number that says how good the engine really
is. Pinnacle **closing** odds (the sharpest public forecast that exists) were
already sitting unused in the raw lake — 106 columns per football-data.co.uk
file, of which we read six.

- ✅ `research/data/odds_loader.py` — closing odds → implied probabilities with
  the bookmaker's **overround removed** (measured at 2.38%, exactly Pinnacle's
  typical margin). Joined to the Understat dataset on (season, home, away); 7
  of 35 club names differ and are mapped explicitly. **Odds are a yardstick,
  never a model input** — the vision's non-goal is upheld.
- ✅ `research/evaluation/benchmark_odds.py` — scores the market as if it were
  just another model, on identical matches and metrics.

**Result** *(run `odds_report_20260710T055226Z`, 7 seasons, 2,660 matches)*

| model | log loss | accuracy | ECE(home) | ECE(draw) |
|---|---|---|---|---|
| **market_closing** (Pinnacle) | **0.9464** | **55.9%** | 0.0249 | 0.0144 |
| **ensemble** (our champion) | **0.9821** | 53.1% | **0.0228** | **0.0074** |
| elo | 0.9857 | 53.2% | 0.0303 | 0.0089 |
| baseline | 1.0650 | 44.1% | 0.0094 | 0.0146 |

- ❌ **The market beats the engine** by 0.0357 log loss (significant, p<0.0001).
  This is the expected, honest outcome — the closing line is the strongest
  public forecast of a football match there is.
- ✅ **Progress metric: the engine has captured 70% of the achievable edge.**
  Baseline → market spans the entire range of publicly-extractable skill; we
  cover 70% of it, and sit **2.8 accuracy points** behind the sharpest
  bookmaker on earth, using only free data.
- ✅ **On calibration we are competitive with the market** — better on home
  (0.0228 vs 0.0249) and markedly better on draws (0.0074 vs 0.0144). The
  market's advantage is *sharpness*, not honesty.
- ⚠️ **A hypothesis we formed here was later REFUTED — see Phase 4c.** We
  reasoned that because the closing line is priced minutes before kickoff (and
  thus knows confirmed line-ups and late team news, which we do not), much of
  the residual gap must be an *information* advantage, and that team news was
  therefore the lever with headroom. **Phase 4c measured it and that is wrong:**
  the entire open→close improvement is worth only 0.0039, while the market's
  *opening* price — which knows no team news either — already beats us by
  0.0318. The gap is a modelling/information advantage present from the start,
  not a timing one. Recorded here rather than deleted, because the reasoning was
  plausible and the measurement is what settled it.

### Phase 4c — Do we hold any EDGE over the market? ✅ *(done — measured: no)*

"Beat the closing line" is too blunt a goal to test. It was decomposed into four
sharper questions, all answered with Pinnacle **opening** and **closing** prices
already in the raw lake. *(Integrity: the T3 blend is a measurement instrument,
never a shipped model. Odds remain a yardstick, not an input.)*

**Result** *(run `market_edge_report_20260710T063543Z`, 2,660 EPL matches)*

| forecast | log loss |
|---|---|
| market_close (Pinnacle, at kickoff) | **0.9464** |
| blend(engine + market_close) | 0.9466 |
| **market_open** (Pinnacle, before team news) | **0.9503** |
| ensemble (our engine) | 0.9821 |

- **T1 — what is late information worth?** open → close improves the market by
  only **0.0039**. Team news *and* sharp money *combined* are worth that much.
- **T2 — can we beat the opening line?** **No.** The opening price, which knows
  no team news, already beats us by **0.0318** (p<0.0001).
- **T3 — do we carry signal the market misses?** **No.** Blending our engine
  into the closing line earns it an average weight of just **4%** and does not
  improve it (0.9466 vs 0.9464). Our forecast is a strict *information subset*
  of the market.
- **T4 — Closing Line Value.** When we disagree with the open, the line moves
  toward us **48.5%** of the time (r = −0.007, p = 0.54) — a coin flip. **We do
  not anticipate the market at all.**

**Verdict: the engine has no measurable edge over Pinnacle, at the open or the
close.** This is the honest, hard-won answer most projects never obtain — they
assume an edge instead of measuring one.

Three consequences, all evidence-based:
1. 🔁 **It refutes the Phase 4a hypothesis** that team news explains the gap
   (only 0.0039 of 0.0357 arrives late), and *explains* why every injury test
   returned "no detectable effect" — the whole channel is small. Two independent
   lines of evidence now agree.
2. 🚫 **No public-data feature is likely to close this gap.** Weather, referees,
   line-ups are all bounded by the same measurement.
3. 🎯 **The only untested route to beating a line** is a *less efficient* market:
   lower divisions (Championship, League One, Scottish, second tiers), which
   football-data.co.uk covers with odds for free, and which our Elo/Dixon-Coles
   models can predict without xG.

### Phase 4d — Beat the line where the market is soft ✅ *(done — hypothesis REFUTED)*

The last live route to beating a closing line: Pinnacle prices the Premier
League with enormous care but charges a fatter margin in the lower divisions
(overround 2.38% in the EPL vs 3.15% in League Two), which reads like a
bookmaker hedging against its own uncertainty. Nine leagues, ~29,800 matches,
matches *and* odds from the same football-data.co.uk file (no cross-source join).
Engine = **Elo** (72% of the champion's weight; within 0.004 of the full ensemble
on the EPL, and it fits in 0.1s).

**Result** *(run `soft_markets_report_20260710T065901Z`)*

| league | overround | engine (elo) | closing line | gap | **edge captured** |
|---|---|---|---|---|---|
| Bundesliga 2 | 2.89% | 1.0723 | 1.0522 | +0.0202 | 26% |
| Championship | 2.59% | 1.0643 | 1.0338 | +0.0304 | 29% |
| **Scottish Prem** | 2.88% | 0.9694 | 0.9314 | +0.0380 | **72%** |
| **Premier League** | 2.38% | 0.9847 | 0.9457 | +0.0390 | **68%** |
| Serie B | 2.88% | 1.0831 | 1.0439 | +0.0392 | 7% |
| League One | 3.10% | 1.0722 | 1.0198 | +0.0524 | 7% |
| **League Two** | 3.14% | 1.0834 | 1.0484 | +0.0350 | **−13%** ⚠️ |

- ❌ **The closing line wins in every league, all p<0.0001. There is no soft
  market.** And the conclusion is robust to the Elo probe: the full ensemble beats
  Elo by only ~0.003 on the EPL — an order of magnitude less than the *smallest*
  gap (0.0202).
- ❌ **The hypothesis is backwards.** Correlation between the bookmaker's margin
  and our gap to its price is **r = +0.18** — where the book is *less* sure, we
  fall *further* behind, not closer.
- 🔑 **Why: a wider margin means genuine unpredictability, not laziness.** The
  whole predictable span (baseline→market) collapses in the lower divisions —
  0.136 in the Scottish Prem, but only 0.031 in League Two. Lower-league football
  is ~4x less predictable **for everyone**, and our model captures far less of the
  little that is there.
- ⚠️ **In League Two, Elo is WORSE than the naive baseline** (1.0834 vs 1.0795):
  the engine has *negative* skill there and must not be used.
- ✅ **Our engine is at its best in the most predictable leagues** — Scottish
  Premiership (72% of the achievable edge, thanks to the Celtic/Rangers gap) and
  the Premier League (68%) — not in the obscure ones.

**Verdict: "beat the closing line" is closed. There is no exploitable betting
edge anywhere we can reach — not at the open, not at the close, not in the lower
leagues.** Definitively measured across 9 leagues and ~30,000 matches, rather
than assumed. The engine's value is therefore *not* betting alpha: it is honest,
calibrated, explainable forecasting, and it is strongest exactly where the
audience is (the top divisions).

*Two real bugs were found and fixed by this phase:* football-data.co.uk publishes
some files in cp1252 (Scottish/Serie B were silently unreadable → `csv_utils.
read_csv_resilient`), and `(season, home, away)` is **not** a unique fixture key
in leagues that split mid-season, where a team hosts the same opponent twice
(→ the date is now part of the join key, with a duplicate guard).

### Phase 4b — Serving + drift *(pending)*

- `prediction-engine/serving/` — FastAPI service returning the scoreline grid,
  derived markets (1X2, over/under, BTTS, correct score), and the explanation.
  *(Note: the champion ensemble currently emits 1X2 only; wiring the scoreline
  grid through the blend is outstanding.)*
- Automated retraining + a **drift monitor** that re-runs the backtest on new
  results and alerts when the champion degrades.
- Repeat the odds benchmark across all five leagues, and refine overround
  removal (Shin's method) to sharpen the market's own estimate.

**Success criterion:** live predictions stay calibrated over a full season, and
the engine measurably closes the gap to the closing-odds benchmark.

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
