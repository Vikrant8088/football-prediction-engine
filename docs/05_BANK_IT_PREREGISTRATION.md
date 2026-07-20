# 05 — Bank It: pre-registration of the 2026/27 live forward test

**Pre-registered 2026-07-16**, before the 2026/27 FPL game opened (~22 Jul 2026)
and well before the GW1 deadline (~21 Aug 2026). No 2026/27 live data existed when
this was written. **Its git commit is the timestamp** — this document, and any change
to it, is dated by the repository history.

---

## Why pre-register

Every number the engine has is in-sample to history. A forward test is only credible
if *what* we measure and *how* we decide are fixed **before** the data exists —
otherwise the result can be retrofitted, consciously or not. This document fixes them.
Any later change is recorded in the [Amendments log](#amendments-log) with a date and
reason; nothing here is edited silently.

---

## Hypothesis (directional)

The engine's fixture-aware projection, used to pick a £100m squad and a captain, will
outscore a `player_ppg` squad and captain in the live 2026/27 Premier League season,
by a per-gameweek margin **consistent with the recent-era backtest (~+3/GW)**. The
pooled eight-season figure is higher (~+5/GW) but is decaying as the market grows
efficient, so the recent end is the honest expectation.

---

## Primary endpoint (single, pre-specified)

- **Unit:** one Premier League gameweek.
- **Our squad:** the provably-optimal legal 15-man squad for **£100.0m**, built from the
  engine's projection (`expected_points`); best legal XI started; highest-projected
  starter captained.
- **Baseline squad:** the identical construction from `player_ppg`.
- **Metric:** mean over scored gameweeks of
  `(our XI actual points) − (baseline XI actual points)`, where each squad's captain
  is doubled — or its **vice-captain** if the captain plays 0 minutes (FPL's rule).
  Both squads lock a captain *and* a vice-captain before the deadline.

This is the **same endpoint validated in `benchmark_fpl_optimizer`** (£100m, with
captain). It is deliberately not the flattering cell: captaincy *reduces* our measured
gain. A single pre-specified endpoint needs no multiplicity correction.

---

## Baseline definition (fixed)

- `player_ppg` = a player's **current-season** points ÷ gameweeks he had a fixture (the
  same denominator the backtest uses).
- **Opening weeks (GW1–5)**, before a stable current-season average exists: **last-season
  ppg**.
- A player with neither: 0 (so the baseline cannot pick him).

---

## Locking procedure (the integrity mechanism)

1. **Before each gameweek's official deadline**, both squads are computed and committed
   to git as `research/results/live/2026-27/GWxx.{json,md}`. The commit predates the
   deadline; git history is the tamper-evident timestamp.
2. Neither squad is **ever recomputed at scoring time**. After the gameweek, actual
   points are looked up and the *locked* squads are scored.
3. A gameweek whose squad is not committed before its deadline is recorded as
   **MISSED, not dropped**. Silently discarding bad weeks is the self-deception this
   project exists to avoid.

---

## Decision rule (fixed)

- **Test statistic:** mean gain/GW over the scored range.
- **Scored range: GW6–38** (33 gameweeks), *not* the full 38. This matches the range the
  backtest actually scored (`FIRST_SCORED_GAMEWEEK = 6`): before GW6 no model has enough
  of the season to say anything, and the live engine's player rates are cold-started
  from last season. Claiming GW1–38 would assert proof over five gameweeks the backtest
  never tested. **GW1–5 are still locked and scored, but reported separately as
  exploratory** (see below) and excluded from the headline.
- **Significance:** paired *t* **and** Wilcoxon signed-rank, **both p < 0.05** (this
  project's two-test rule), evaluated over that range.
- **Interim:** a single read at **GW19**, reported and explicitly labelled *INTERIM*. It
  is not the verdict, and its p-values are not corrected for the look.
- **Success:** mean gain/GW > 0 **and** both tests pass → the forward edge is confirmed.
- **Null:** mean gain/GW ≤ 0 → the backtest was optimistic; recorded as such, not buried.
- **Power caveat (stated up front):** one season is ~33 paired points, and gameweek gains
  have a large spread (SD ≈ 15–20 pts in backtest). A true ~+3/GW edge therefore has
  **low power to reach p < 0.05 in a single season** — as the 2025/26 dry-run of the
  pipeline showed (+5.24/GW yet p ≈ 0.07). So the season outcome is **evidence weighed
  together with the eight-season backtest, not a lone verdict**: a positive-but-
  nonsignificant season matching the backtested magnitude *corroborates*; a
  zero-or-negative season *contradicts*.

---

## Declared variant: the predicted-lineup candidate (secondary, pre-specified)

Phase 6f measured a large headroom for minutes information (+4.59/GW directional,
8/8 seasons), and predicted lineups are the only pre-deadline route to a slice of it.
That candidate **cannot be backtested** — nobody archived predicted XIs — so it can
only be tested forward, here.

- **The primary endpoint above is unchanged.** It remains the *proven* recent-form
  champion with **no** lineup feed. An unproven candidate does not get to be the
  headline.
- Each gameweek, a **second squad** is locked alongside it, identical in every respect
  except that its minutes model also consumes the pre-deadline `Start %`.
- **Variant metric:** mean points/GW of the lineup squad **minus the primary squad**,
  paired by gameweek. Same gameweek, same prices, same deadline, same actuals — so the
  difference is the lineup signal and nothing else.
- Same two-test rule (paired *t* **and** Wilcoxon, both p<0.05) and the same power
  caveat: one season is unlikely to settle it alone.
- Scored **only on gameweeks where the variant was actually locked**, so a week with a
  missing feed cannot let it borrow the primary's result.
- **Expectation, stated up front:** a *fraction* of +4.59, not the whole thing. The
  live engine already uses FPL's injury flags, so the marginal prize is **rotation
  only** — who the manager picks among *fit* players.

Locking both is what makes this honest: simply switching the feed on would confound
the lineup signal with everything else and leave us unable to say whether it helped.

## Declared variant: the carried-squad transfer A/B (secondary, pre-specified)

Declared **2026-07-20**, before the season opened and before any live gameweek was
locked. This is the "separate, separately pre-registered test" that the *Fixed
procedural choices* section below anticipated.

**Why it exists.** `benchmark_fpl_transfers` (4 seasons, 131 paired gameweeks)
established two things and failed to establish a third:

| | result |
|---|---|
| Acting beats holding | **+7.17 net pts/GW**, both tests, positive 4/4 seasons — **proven** |
| Blind random transfers | **29.88/GW** vs 44.28 for holding — so the gain is not an artifact of a rotting null |
| **Our projection beats `player_ppg` at *choosing* transfers** | **−1.56/GW**, *t* p=0.31, W p=0.34 — **a null, directionally negative** |

The third is the open question, and it is the one that decides how the live team
should actually be run. 131 gameweeks did not settle it, so it goes to the season.

**The two tracks.** Both carry a squad and run the *same* policy — one free transfer
a week, made whenever it improves the projected XI, **never taking a hit** (the
policy the backtest pre-specified, not a tuned one):

- `carried_ours` — maintained by our projections
- `carried_ppg` — maintained by `player_ppg`

**Both open from the identical fifteen** (the primary's opening squad), so from the
first transfer onward the only difference between them is *which number chose it*.

- **Variant metric:** mean net points/GW of `carried_ours` **minus** `carried_ppg`,
  paired by gameweek, hits subtracted. Same two-test rule.
- Scored **only on gameweeks where both tracks were locked**.
- **The primary endpoint is untouched.** It still rebuilds a fresh £100m squad every
  gameweek — that is what the 8-season proof describes, and neither track is allowed
  to become the headline.
- **Direction stated up front:** the backtest points *against* `carried_ours`. If the
  season agrees, the honest conclusion is that our projections should not drive
  transfers, and that must be reported as readily as a win would be.

**Extra integrity control.** A carried squad is *stateful*, which the rebuild endpoint
is not — so a state file edited after kickoff would silently destroy the test while
still looking like a locked prediction. Two mitigations: state is written to
`research/results/live/<season>/` and committed **before** each deadline, and
`carried.step` **refuses to re-decide a gameweek it has already decided**.

## Secondary / context metrics (reported, never pass/fail)

- **Overall rank percentile** among all FPL managers, if the squad is entered as a real
  team — the ultimate external check, but noisy over one season.
- **Template** (most-owned XI) points, as the "just follow the crowd" reference.

---

## Fixed procedural choices (v1)

- **Rebuild-from-scratch** each gameweek — no carried squad, no transfers or hits. This
  matches the backtest that produced the proven number. A transfer-realistic variant, if
  built, is a **separate, separately pre-registered** test — now built and declared
  above (*the carried-squad transfer A/B*, 2026-07-20). **The primary remains
  rebuild-from-scratch**; the carried tracks sit alongside it and never replace it.
- **Opening-week cold start (GW1–5):** FPL zeroes every season-to-date stat between
  seasons, so on opening day `xg_per_90` is 0 for **everyone** — Haaland would project
  exactly as any other nailed-on striker (1.95 xPts: appearance points only). For GW1–5
  the engine therefore substitutes **last season's per-90 rates**, joined on FPL's
  permanent player `code`, for players who have yet to feature. This is a cold start,
  not an override: a player with real current-season minutes keeps his own numbers.
  It is **unvalidated by construction** — the backtest never scored GW1–5 — which is
  precisely why those gameweeks are reported as exploratory and excluded from the
  headline. Known blind spot: a summer signing with no previous Premier League season
  has no rates at all and stays invisible until he plays.
- **Vice-captain:** each squad locks a captain and a vice-captain (the 2nd-highest-
  projected starter). If the captain plays 0 minutes, the vice-captain is doubled
  instead; if both play 0, nobody is doubled. Applied identically to both squads.
- **No outfield autosubs:** a locked starter (other than the captain→vice case) who
  plays 0 minutes scores 0. Applied identically to both squads.
- **The real game:** budget exactly £100.0m, quota 2/5/5/3, max 3 per club.

---

## Known deviations from the backtest (honesty)

- **xG source:** the live projection uses FPL/Opta season-to-date xG; the eight-season
  proof used Understat walk-forward xG. Same *method*, different *source* (FPL published
  no xG before 2022/23). Recorded in each artifact's `config`.
- **Injury flags:** the live projection uses FPL availability flags, which the historical
  backtest could not — this should *help* the live projection relative to the backtested
  figure.
- **Scoring rules:** the engine models the 2025/26 rules. **If 2026/27 changes them,
  `scoring.py` is updated — and that update logged — before GW1** (milestone M3). A proof
  run on stale rules is void.

---

## Amendments log

Any change to this protocol after publication is recorded here with date and reason.

- **2026-07-16 — vice-captain rule added.** The original draft doubled the captain
  unconditionally. FPL promotes the vice-captain when the captain plays 0 minutes, so
  each squad now locks a vice-captain and the scorer applies the promotion. Made before
  any live data existed; applies identically to both squads, so it does not bias the
  comparison — it only makes the scoring match the real game. *(Reason: correctness.)*

- **2026-07-17 — predicted-lineup variant declared** (see the section above). Each
  gameweek now locks a *third* squad whose minutes model consumes the pre-deadline
  `Start %`, tested against the primary as a paired A/B. **The primary endpoint is
  untouched** and still uses no lineup feed — this adds a separately-declared secondary
  test rather than altering the pre-registered one. Declared before any live data
  existed. *(Reason: Phase 6f showed a large minutes headroom, and the candidate cannot
  be backtested — locking both variants is the only way to attribute the effect instead
  of confounding it with a silent switch.)*

- **2026-07-17 — scored range corrected to GW6–38, and opening-week rates cold-started.**
  This document originally claimed a "full 38-gameweek season" horizon. That was an
  error on my part: the backtest scores from GW6 (`FIRST_SCORED_GAMEWEEK = 6`), so the
  proven +3/GW describes **33** gameweeks, not 38, and the original wording would have
  asserted proof over five gameweeks nobody ever tested. Separately, FPL zeroes all
  season-to-date stats between seasons, so without last season's rates the GW1 squad
  would have been picked on appearance points alone (Haaland 1.95 xPts, identical to
  every other nailed striker — measured, not supposed). Both fixed: the primary is now
  scored GW6–38, GW1–5 are locked and reported as **exploratory**, and opening-week
  rates are cold-started from last season. Declared before any live data existed.
  *(Reason: the horizon overstated the proof, and the engine was broken in the exact
  gameweeks the horizon covered.)*

- **2026-07-17 — last-season ppg implemented.** The GW1–5 baseline rule above was
  pre-registered but never coded; at GW1 every player tied on zero, which would have
  made the baseline squad degenerate and the comparison meaningless on day one. Now
  computed, joined on FPL's permanent player `code` (element ids are reassigned each
  summer, so an id join would credit a player with another's history). *(Reason: the
  code did not keep a promise this document had already made.)*

- **2026-07-20 — carried-squad transfer A/B declared** (see the section above). Two
  further tracks are locked each gameweek, `carried_ours` and `carried_ppg`, opening
  from the primary's fifteen and differing only in which projection chooses the weekly
  transfer. **The primary endpoint is untouched** and still rebuilds from scratch.
  Declared before the season opened and before any live gameweek was locked. *(Reason:
  the transfer backtest proved acting beats holding (+7.17 net pts/GW) but returned a
  null, directionally negative, on whether OUR projections choose better transfers than
  a season average (−1.56/GW, t p=0.31). That question decides how the live team is
  actually run and 131 backtest gameweeks did not settle it, so it goes to the season
  as a paired A/B rather than being resolved by assumption. The direction the evidence
  currently points — against `carried_ours` — is stated up front so a loss cannot be
  quietly reframed later.)*
