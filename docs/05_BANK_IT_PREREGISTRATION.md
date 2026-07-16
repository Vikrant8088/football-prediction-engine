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
  `(our XI actual points, captain doubled) − (baseline XI actual points, captain doubled)`.

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

- **Test statistic:** mean gain/GW over the season.
- **Significance:** paired *t* **and** Wilcoxon signed-rank, **both p < 0.05** (this
  project's two-test rule), evaluated over the **full 38-gameweek season**.
- **Interim:** a single read at **GW19**, reported and explicitly labelled *INTERIM*. It
  is not the verdict, and its p-values are not corrected for the look.
- **Success:** mean gain/GW > 0 **and** both tests pass → the forward edge is confirmed.
- **Null:** mean gain/GW ≤ 0 → the backtest was optimistic; recorded as such, not buried.
- **Power caveat (stated up front):** one season is ~38 paired points, and gameweek gains
  have a large spread (SD ≈ 15–20 pts in backtest). A true ~+3/GW edge therefore has
  **low power to reach p < 0.05 in a single season** — as the 2025/26 dry-run of the
  pipeline showed (+5.24/GW yet p ≈ 0.07). So the season outcome is **evidence weighed
  together with the eight-season backtest, not a lone verdict**: a positive-but-
  nonsignificant season matching the backtested magnitude *corroborates*; a
  zero-or-negative season *contradicts*.

---

## Secondary / context metrics (reported, never pass/fail)

- **Overall rank percentile** among all FPL managers, if the squad is entered as a real
  team — the ultimate external check, but noisy over one season.
- **Template** (most-owned XI) points, as the "just follow the crowd" reference.

---

## Fixed procedural choices (v1)

- **Rebuild-from-scratch** each gameweek — no carried squad, no transfers or hits. This
  matches the backtest that produced the proven number. A transfer-realistic variant, if
  built, is a **separate, separately pre-registered** test.
- **No autosubs:** a locked starter who plays 0 minutes scores 0. Applied identically to
  both squads.
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

- *(none — as pre-registered 2026-07-16)*
