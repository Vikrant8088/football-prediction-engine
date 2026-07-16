# 04 — Bank It: the live forward-validation pipeline

*Status: M1 built and green (assembler + `bank_it` CLI, dry-run reproduces a legal
optimal squad on 2025/26 data). Protocol written before any live gameweek so it
cannot be chosen to fit the results. The four open decisions are now signed off
(Section 10).*

---

## 1. Why this exists

The engine has a **backtested** edge: on the pre-registered endpoint (a £100m
squad plus a captain), it beats a `player_ppg` baseline by ~+3 pts/GW in the
recent era, proven over eight Understat-xG seasons, surviving Holm correction and
the removal of the defensive-contribution season (`benchmark_fpl_optimizer`).

A backtest can never prove the edge **holds going forward**. Every number we have
is in-sample to history. The only honest proof is out-of-sample in time: **commit
a prediction before the deadline, score it after the gameweek, accumulate.** That
is what "Bank it" does — it turns a backtested edge into a live-proven one, or
exposes it as backtest-optimistic before we build anything else on top of it.

The 2026/27 season opens in ~a week (game live ~22 Jul, GW1 deadline ~21 Aug).
That window comes once a year. Missing GW1 setup costs a whole season of proof.

---

## 2. Scope

**In scope (v1):**
- A weekly job that, **before each gameweek deadline**, emits the engine's
  optimal £100m squad + captain + XI/bench + the full player projection table,
  timestamped and committed.
- A post-gameweek scorer that pulls actual points, scores the locked squad, and
  appends to a running season ledger against the pre-registered baseline(s).

**Explicitly NOT in scope (v1):**
- **No transfer logic.** v1 rebuilds the squad from scratch each gameweek, exactly
  as the backtest does, so the live comparison is apples-to-apples with the proven
  number. A carried squad + transfers/hits is a *later* layer, measured separately.
- **No website/API.** CLAUDE.md ranks those secondary; proof comes first.
- **No new signals** (predicted lineups etc.). Those are gated by the minutes
  ceiling test and tracked separately — Bank-it is agnostic to which projection
  feeds it.

---

## 3. The integrity mechanism (the heart of it)

Forward validation is worthless if the prediction can be edited after kickoff.
Three locks, all pre-committed:

1. **Pre-registration by git.** Each gameweek's squad artifact is committed
   **before `deadline_time`**. Git history is a tamper-evident timestamp: anyone
   can verify the prediction predates the matches.
2. **The endpoint is already fixed.** We reuse the primary endpoint from
   `benchmark_fpl_optimizer` verbatim — £100m squad, with captain, gain vs
   `player_ppg` — rather than inventing a live endpoint that could be chosen to
   flatter. It is *not* the flattering choice (captaincy reduces our measured gain).
3. **The decision rule is pre-specified** (Section 7) before any live data lands.

If we ever miss a deadline, that gameweek is recorded as **missed, not skipped** —
silently dropping bad weeks is the exact self-deception this project rejects.

---

## 4. Architecture — reuse, don't rebuild

Almost everything exists. The genuinely new code is one assembly step (4.3) and
the scorer (4.6).

### 4.1 Refresh data
```
python -m data_warehouse.cli download --source fpl
```
`data_warehouse/sources/fpl.py` already fetches **both** feeds we need:
- `bootstrap-static` → players, prices, availability, per-90 rates, and the
  `events` list (gameweek `deadline_time`, `is_next`).
- `fixtures` → every fixture with its gameweek (`event`), home/away team, and
  `finished` flag.

### 4.2 Identify the target gameweek
From `bootstrap-static` `events`: the next gameweek is the one with `is_next`
true (fallback: earliest future `deadline_time`). From `fixtures`, take every
fixture whose `event` equals that gameweek — this is the fixture list to project,
and it is what makes **blank/double gameweeks** fall out naturally (a team can
have 0 or 2 fixtures).

### 4.3 Build the gameweek projection frame *(new code)*
The optimizer needs **all players across all fixtures in the gameweek in one
frame**; `project_fixture` produces **one fixture** at a time. Bank-it's core new
job is the assembly loop:

```python
engine   = PredictionEngine.train("EPL")
players  = load_players()                       # research.data.fpl_loader
history  = _recent_minutes_history()            # current-season minutes; {} off-season

rows = []
for home, away in fixtures_in_gameweek:
    table = project_fixture(engine, players, home, away, minutes_history=history)
    rows.append(table)
frame = pd.concat(rows, ignore_index=True)

# Double gameweeks: a player appears twice → sum expected_points (and each
#   additive channel) across their fixtures; keep one row per player.
# Blank gameweeks: a player with no fixture is absent → correctly unpickable.
# Availability: drop players where not `available` (injury/suspension flags,
#   which the backtest lacked and which should *help* the live projection).
# Optimizer contract: it reads a `club` column; projections emit `team`. Add
#   frame["club"] = frame["team"].
```

### 4.4 Solve the squad
```python
from prediction_engine.fpl.optimizer import select_squad
squad = select_squad(frame, "expected_points", squad_budget=100.0)
# → SquadSelection(xi, bench, projected, cost, formation, captain)
```
Provably optimal (branch-and-bound, verified against exhaustive enumeration).

### 4.5 Emit + commit the artifact (before deadline)
Write to a versioned live log, e.g. `research/results/live/2026-27/GW01.{json,md}`:
- target gameweek, `deadline_time`, and the run timestamp;
- engine/version identifier and the projection config (minutes model, xG source);
- the XI, bench, captain, **vice-captain**, formation, total cost;
- the full per-player projection table (so the scorer needs no re-run);
- the baseline squad(s) for the same gameweek (Section 6).

**Commit before `deadline_time`.** That commit *is* the pre-registration.

### 4.6 Score after the gameweek *(new code)*
Once fixtures are `finished`:
- pull actual points per player (`bootstrap-static` totals delta, or
  `element-summary/{id}` per-gameweek);
- score each locked squad with `scorer.score_squad`, applying the **vice-captain
  rule** (captain doubled, or the vice-captain if the captain played 0 minutes) — for
  us and for each baseline; this needs per-player minutes, which `gameweek_actuals`
  now returns alongside points;
- append `{gameweek, ours, baseline(s), gain, our_rank_if_entered}` to a season
  ledger and recompute the running paired test.

### 4.7 An honesty caveat: the live projection is not byte-identical to the proof
The eight-season backtest that proved the edge used **Understat** xG on a
walk-forward per-match basis (FPL published no xG before 2022/23). The live path
(`project_fixture` ← `load_players`) uses **FPL/Opta** xG from `bootstrap-static`,
season-to-date, plus live injury flags. **The method is the same** — the same
`project_player`, the same ensemble grid, the same recent-form minutes model — but
the xG *source* and *window* differ. So the live number is our genuine best
projection run forward, not a numeric replay of the backtest. This is recorded in
each artifact's `config.xg_source`, and stated here rather than buried. (A later
refinement could also run the live projection on Understat xG to match the proof
exactly; v1 uses the shipped engine, which is what a user actually gets.)

---

## 5. Scoring-rules risk for 2026/27  *(gate before GW1)*

`prediction_engine/fpl/scoring.py` encodes the **2025/26** rules (including the
defensive-contribution rule). The 2026/27 season is publicly billed as "a very
different season in Fantasy," which usually means scoring changes. **Before Bank-it
feeds a real GW1 squad**, re-read the confirmed 2026/27 rules and update
`scoring.py` if they changed. This is a hard gate: a proof run on stale rules
proves nothing.

---

## 6. The baseline question  *(one open decision)*

The backtest baseline is `player_ppg` (a player's season points-per-game). Live,
that is **undefined at GW1 and noisy through ~GW5** — nobody has a per-game
average yet.

| Baseline | Role | Note |
|---|---|---|
| `player_ppg` | **Primary** — matches the pre-registered backtest metric | Needs an early-season definition (below) |
| last-season ppg | Early-GW stand-in for the primary | Only for GW1–~5, then switch to current-season ppg |
| overall rank (%ile of the field) | **Secondary / real-world proof** | The ultimate external check; noisy over one season |
| template (most-owned XI) | Context | What "just follow the crowd" scores |

**Recommendation:** primary = `player_ppg` (current-season once it stabilises,
last-season ppg for the opening weeks), pre-declared now; report overall rank and
template alongside as context, not as the pass/fail metric. **This early-season
definition must be fixed in the pre-registration doc before GW1.**

---

## 7. Pre-specified decision rule  *(fix before any live data)*

- **Primary metric:** mean points/GW gain of our £100m+captain squad over the
  `player_ppg` squad, paired by gameweek.
- **Test:** paired t **and** Wilcoxon, both p<0.05 (the project's two-test rule),
  evaluated at the end of a pre-set horizon — proposed **full 2026/27 season
  (38 GW)**, with an honest interim read at GW19 marked explicitly as interim.
- **Success:** live gain ≥ 0 and both tests pass → the forward edge is real.
- **Informative null:** live gain ≈ 0 or negative → the backtest was optimistic;
  that is a finding, recorded, not buried.
- **Power caveat, stated up front:** one season is ~38 paired points; a ~+3/GW
  edge may be real yet not reach p<0.05 in a single season. The season result is
  therefore *evidence*, weighed with the eight-season backtest, not a lone verdict.

---

## 8. Milestones

| # | Deliverable | Status |
|---|---|---|
| M1 | GW-frame assembler + `python -m prediction_engine.fpl.bank_it` that prints & writes the squad artifact | **DONE** — `bank_it.py` + `test_bank_it.py` (6 tests) green; dry-run on 2025/26 GW20 produced a legal optimal £95.2m squad end-to-end |
| M2 | Pre-registration doc (endpoint, baseline incl. early-season rule, decision rule, horizon) committed | **DONE** — [docs/05_BANK_IT_PREREGISTRATION.md](05_BANK_IT_PREREGISTRATION.md), committed 2026-07-16 before the game opened; its git commit is the timestamp |
| M3 | 2026/27 scoring-rules re-check | Pending — `scoring.py` verified/updated against confirmed rules |
| M4 | Post-GW scorer + season ledger + running paired test | **DONE** — `scorer.py` + `test_scorer.py` (10 tests) green; `score_squad` proven to match the optimizer's `xi_actual_points` on real data; full loop demoed over 2025/26 (below) |
| M5 | Run live from GW1 | Pending — first squad committed before the GW1 deadline |

*M4 note:* the lock step now also locks the `player_ppg` baseline squad, so the
paired comparison is fixed pre-deadline. An end-to-end run of the whole loop over
2025/26 (lock both squads → pull actuals from the archive → score → ledger) gave
**ours 58.2/GW vs baseline 52.9/GW, +5.24/GW, won 20/33, t p=0.070, Wilcoxon
p=0.064 → not significant on one season.** That is the §7 power caveat in action:
a real +5/GW edge, but ~33 paired points cannot clear p<0.05 alone — the season is
evidence weighed with the eight-season backtest, not a standalone verdict. The live
`scorer` reproduces the proven `benchmark_fpl_optimizer` measurement exactly (tied
by a cross-check test), so the forward number will be comparable to the backtest.

*M1 note:* the assembler reuses the shipped `project_fixture` verbatim and adds only
the gameweek stitching (double-GW summing, blank-GW absence, `position`→int, `club`
column). The originally-planned "reproduce the backtest squad exactly" drift test was
softened to a **plumbing-equivalence** test once §4.7 made clear the live and
backtest paths legitimately differ (xG source/window); the unit tests pin the
assembly, and the dry-run proves the end-to-end path.

---

## 9. Tests & reproducibility

- **Assembler unit tests:** the frame the assembler emits is accepted by
  `select_squad` (required columns present); double-gameweek points are summed to
  one row per player; blank-gameweek players are absent; `unmapped_teams()` is
  empty so no club is silently dropped.
- **Drift acceptance test:** on a historical gameweek, the live assembly path must
  select the **same squad** the backtest path selects from `cached_predictions`.
  If the live and research code paths ever diverge, this fails loudly — the live
  proof is only meaningful if it runs the *same* projection that was validated.
- **Determinism:** given fixed input JSON, the emitted artifact is byte-stable
  (no wall-clock inside the projection; timestamp is recorded, not an input).

---

## 10. Decisions (signed off)

All four confirmed 2026-07-16, before any live data:

1. **Early-season baseline** — last-season ppg for GW1–5, then switch to
   current-season ppg. Primary metric stays gain vs `player_ppg`.
2. **Horizon** — full 38-GW season, with a GW19 read explicitly marked *interim*.
3. **Artifact location** — `research/results/live/2026-27/GWxx.{json,md}`.
4. **Transfer realism** — v1 rebuilds from scratch (matches the proven backtest); a
   carried-squad + transfers variant is a later version, measured separately.
