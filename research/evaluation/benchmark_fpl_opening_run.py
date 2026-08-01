"""Does weighting the OPENING squad over its fixture RUN beat picking on one gameweek?

The engine picks the single-gameweek-optimal squad. That is correct for the
pre-registered measurement (which rebuilds every week), but an opening squad you
actually HOLD should arguably weight the first several fixtures: you cannot cheaply
transfer out of a bad early run, and we measured that our projection does not out-pick
a season average at the transfer margin (a null). So getting the opening run right may
matter more than fixing it later.

This tests it honestly. At each start gameweek G it builds two 15-man squads and HOLDS
each for N gameweeks (no transfers), fielding the best XI each week and scoring on
ACTUAL points:

    single  maximise the projection for gameweek G only (what the engine does today)
    run     maximise a decay-weighted projection over G .. G+N-1

LEAKAGE CONTROL — the load-bearing part. The `run` squad may only use information
available at G. The team model is trained on matches BEFORE G's kickoff (exactly as the
walk-forward backtest trains it), and every future fixture G+k is projected with that
same as-of-G model, as-of-G player rates, and as-of-G minutes — the ONLY thing read
ahead of time is which opponent each team faces, which is legitimately on the published
schedule. Nothing about how those future matches actually turned out touches the pick.

Windows are NON-OVERLAPPING (G = 6, 6+N, 6+2N, …) so the held-point sums are independent
rather than a smear of the same gameweeks, which keeps the paired test honest (and the
sample small — reported as effect size first, p-value second).

Pre-specified before the numbers existed: PRIMARY = `run` (decay 0.85, N=5) vs `single`,
paired by (season, window), two-test rule (paired t AND Wilcoxon, both p<0.05).
Sensitivity over decay and N, Holm-corrected. A null here means "pick on GW1, the
opening run does not need special weighting" — a perfectly good answer to have measured.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.manager import best_xi
from prediction_engine.fpl.optimizer import select_squad
from research.data.fpl_archive import load_gameweeks
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_projections import (
    FIRST_SCORED_GAMEWEEK, MIN_TRAINING_SEASONS, SEASONS_WITH_XG, _GridCache,
    _project_gameweek, _rates_from_history, team_scoring_rates)
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from prediction_engine.fpl.minutes import recent_form_minutes

logger = logging.getLogger(__name__)

PRIMARY_DECAY = 0.85
PRIMARY_N = 5
SENSITIVITY = [(1.0, 5), (0.70, 5), (0.85, 3), (0.85, 8)]


def season_windows(season, matches, hold, minutes_half_life=2.0):
    """Non-overlapping held windows for one season.

    Each window is a dict of per-player as-of-G projections over the next `hold`
    gameweeks plus the actual points they scored, enough to build and score either
    squad. Everything in the projection is as-of-G; only the fixture opponents are read
    ahead (the published schedule).
    """
    frame = load_gameweeks(season)
    fixtures_by_gw = defaultdict(lambda: defaultdict(list))
    for row in frame.to_dict("records"):
        fixtures_by_gw[row["gameweek"]][row["player_id"]].append(row)
    positions = frame.groupby("player_id")["position"].first().to_dict()
    prices_by_gw = {(r["gameweek"], r["player_id"]): float(r["price"])
                    for r in frame.to_dict("records")}
    clubs = frame.groupby("player_id")["team"].first().to_dict()

    gameweeks = sorted(fixtures_by_gw)
    history = defaultdict(list)
    gw_totals = defaultdict(list)
    minutes_history = defaultdict(list)

    windows = []
    next_start = FIRST_SCORED_GAMEWEEK       # non-overlapping cursor

    for elapsed, gameweek in enumerate(gameweeks):
        by_player = fixtures_by_gw[gameweek]

        # A window opens only at the cursor, only if `hold` more gameweeks remain.
        if gameweek == next_start and gameweeks.index(gameweek) + hold <= len(gameweeks):
            window_gws = gameweeks[gameweeks.index(gameweek):
                                   gameweeks.index(gameweek) + hold]
            kickoff = min(f["kickoff_time"] for fs in by_player.values() for f in fs)
            cutoff = pd.Timestamp(kickoff).tz_convert(None)
            train = matches[matches["date"] < cutoff]
            if train["season"].nunique() >= MIN_TRAINING_SEASONS:
                model = _GridCache(ScorelineEnsemble().fit(train))
                rates = team_scoring_rates(train)
                players = {}
                for player_id in set().union(*[set(fixtures_by_gw[g]) for g in window_gws]):
                    prior = gw_totals[player_id]
                    if not prior:
                        continue                 # no history yet -> unprojectable at G
                    hist_rates = _rates_from_history(history[player_id], elapsed)
                    minutes_model = recent_form_minutes(
                        minutes_history[player_id], half_life_matches=minutes_half_life)
                    proj, actual = [], []
                    for g in window_gws:
                        fx = fixtures_by_gw[g].get(player_id, [])
                        if fx:
                            p = _project_gameweek(model, rates, fx, hist_rates,
                                                  positions[player_id],
                                                  minutes_model=minutes_model)
                            proj.append(0.0 if p is None else float(p))
                            actual.append(float(sum(f["total_points"] for f in fx)))
                        else:
                            proj.append(0.0)     # blank gameweek: no fixture, no points
                            actual.append(0.0)
                    players[int(player_id)] = {
                        "proj": proj, "actual": actual,
                        "position": int(positions[player_id]),
                        "club": clubs[player_id],
                        # Price at the window's first gameweek (no season-end leak).
                        "price": prices_by_gw.get((gameweek, player_id), 5.0),
                    }
                if players:
                    windows.append({"season": season, "start": gameweek,
                                    "hold": hold, "players": players})
            next_start = gameweeks[min(gameweeks.index(gameweek) + hold,
                                       len(gameweeks) - 1)]

        for player_id, fixtures in by_player.items():
            history[player_id].extend(fixtures)
            gw_totals[player_id].append(sum(f["total_points"] for f in fixtures))
            minutes_history[player_id].append(sum(f["minutes"] for f in fixtures))

    return windows


def _squad_frame(players):
    return pd.DataFrame([
        {"player_id": pid, "position": p["position"], "club": p["club"],
         "price": p["price"], "value": 0.0}
        for pid, p in players.items()]).set_index("player_id", drop=False)


def _build_squad(players, weights):
    """The provably-optimal £100m squad by the (weighted) projection in `weights`."""
    frame = _squad_frame(players)
    frame["value"] = [float(np.dot(players[pid]["proj"], weights))
                      for pid in frame["player_id"]]
    squad = select_squad(frame, "value", squad_budget=100.0)
    if squad is None:
        return None
    return [int(frame.loc[i, "player_id"]) for i in list(squad.xi) + list(squad.bench)]


def _held_points(squad_ids, players, hold):
    """Actual points from HOLDING this squad `hold` weeks: each week field the best XI
    by that week's as-of-G projection (captain doubled), score on actuals."""
    total = 0.0
    for w in range(hold):
        proj_w = {pid: players[pid]["proj"][w] for pid in squad_ids}
        positions_w = {pid: players[pid]["position"] for pid in squad_ids}
        choice = best_xi(squad_ids, proj_w, positions_w)
        if choice is None:
            continue
        actual_w = {pid: players[pid]["actual"][w] for pid in squad_ids}
        total += sum(actual_w[pid] for pid in choice.xi) + actual_w[choice.captain]
    return total


def compare(windows, decay, hold):
    weights_run = [decay ** k for k in range(hold)]
    weights_single = [1.0] + [0.0] * (hold - 1)

    single_pts, run_pts, rows = [], [], []
    for w in windows:
        if w["hold"] != hold:
            continue
        sq_single = _build_squad(w["players"], weights_single)
        sq_run = _build_squad(w["players"], weights_run)
        if sq_single is None or sq_run is None:
            continue
        s = _held_points(sq_single, w["players"], hold)
        r = _held_points(sq_run, w["players"], hold)
        single_pts.append(s)
        run_pts.append(r)
        rows.append({"season": w["season"], "start": w["start"],
                     "single": s, "run": r, "gain": r - s})

    a, b = np.array(run_pts, float), np.array(single_pts, float)
    diff = a - b
    n = len(diff)
    t_p = float(stats.ttest_rel(a, b)[1]) if n > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1]) if n > 0 else float("nan")
    except ValueError:
        w_p = float("nan")
    return {
        "decay": decay, "hold": hold, "windows": n,
        "run_per_window": float(a.mean()) if n else float("nan"),
        "single_per_window": float(b.mean()) if n else float("nan"),
        "mean_gain_per_window": float(diff.mean()) if n else float("nan"),
        "mean_gain_per_gw": float(diff.mean() / hold) if n else float("nan"),
        "windows_won": int((diff > 0).sum()),
        "paired_t_p": t_p, "wilcoxon_p": w_p,
        "significant": bool(n > 2 and diff.mean() > 0 and t_p < 0.05 and w_p < 0.05),
        "rows": rows,
    }


def apply_holm(cells):
    count = len(cells)
    for test in ("paired_t_p", "wilcoxon_p"):
        order = sorted(range(count), key=lambda i: cells[i][test])
        ok = True
        for rank, i in enumerate(order):
            ok = ok and cells[i][test] < 0.05 / (count - rank)
            cells[i][test + "_survives"] = ok
    for c in cells:
        c["significant_corrected"] = bool(
            c["mean_gain_per_window"] > 0 and c["paired_t_p_survives"]
            and c["wilcoxon_p_survives"])


def build_report(primary, cells, run_id):
    lines = [
        "# FPL opening run: weight the fixture run, or pick on GW1? — %s" % run_id,
        "",
        "Hold a squad for N gameweeks, no transfers, fielding the best XI each week and "
        "scoring on actual points. `single` picks it on the first gameweek's projection; "
        "`run` picks it on a decay-weighted projection over the whole hold. Everything "
        "the `run` squad sees is **as of the start gameweek** — only the fixture "
        "schedule is read ahead, never how those matches turned out.",
        "",
        "## Primary (pre-specified): `run` (decay %.2f, N=%d) vs `single`" % (
            PRIMARY_DECAY, PRIMARY_N),
        "",
        "> **%+.2f points per window** (%.1f vs %.1f over %d-GW holds), "
        "**%+.2f/GW**, winning %d/%d windows." % (
            primary["mean_gain_per_window"], primary["run_per_window"],
            primary["single_per_window"], primary["hold"], primary["mean_gain_per_gw"],
            primary["windows_won"], primary["windows"]),
        ">",
        "> paired t p=%.4f · Wilcoxon p=%.4f → **%s** the two-test rule." % (
            primary["paired_t_p"], primary["wilcoxon_p"],
            "PASSES" if primary["significant"] else "FAILS"),
        "",
        "Windows are non-overlapping, so this is only %d independent holds across the "
        "seasons — low power. Read the effect size first." % primary["windows"],
        "",
        "## Sensitivity (Holm-corrected)",
        "",
        "| decay | N | gain/window | gain/GW | won | t p | Wilcoxon p | both? | corrected |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append("| %.2f | %d | **%+.2f** | %+.2f | %d/%d | %.4f | %.4f | %s | %s |" % (
            c["decay"], c["hold"], c["mean_gain_per_window"], c["mean_gain_per_gw"],
            c["windows_won"], c["windows"], c["paired_t_p"], c["wilcoxon_p"],
            "✅" if c["significant"] else "❌",
            "**✅**" if c.get("significant_corrected") else "❌"))

    survivors = sum(1 for c in cells if c.get("significant_corrected"))
    verdict = ("**Weighting the opening run helps.** " if primary["significant"]
               else "**Null: picking on GW1 is enough.** ")
    lines += [
        "",
        "## Verdict",
        "",
        verdict + ("The pre-specified `run` squad beats `single` by %+.2f/GW held over "
                   "the opening weeks, clearing both tests; %d/%d sensitivity cells "
                   "survive correction. Worth weighting the opening squad over its "
                   "fixture run." % (primary["mean_gain_per_gw"], survivors, len(cells))
                   if primary["significant"] else
                   "Over %d independent holds the opening-run weighting gains only "
                   "%+.2f/GW (t p=%.4f, W p=%.4f) — not enough to clear the bar. On this "
                   "evidence the single-gameweek squad is fine for the opener; the "
                   "fixture run does not need special weighting, likely because strong "
                   "teams tend to be strong across the run anyway and the single-GW pick "
                   "already captures most of it." % (
                       primary["windows"], primary["mean_gain_per_gw"],
                       primary["paired_t_p"], primary["wilcoxon_p"])),
        "",
        "## Caveats",
        "- Non-overlapping windows -> small sample -> low power; effect size leads.",
        "- Held with NO transfers, to isolate the opening-squad choice. A real manager "
        "transfers, which would blunt any opening-run advantage further.",
        "- Both squads field the best XI each week by the as-of-G projection, so the "
        "comparison is purely the 15 players chosen.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(log_dir=Path(__file__).resolve().parents[2] / "logs",
                      filename="benchmark_fpl_opening_run.log", level="INFO",
                      max_bytes=5 * 1024 * 1024, backup_count=3)
    matches = load_understat_matches("EPL")

    holds = sorted({PRIMARY_N} | {n for _, n in SENSITIVITY})
    windows_by_hold = {}
    for hold in holds:
        w = []
        for season in SEASONS_WITH_XG:
            w.extend(season_windows(season, matches, hold))
        windows_by_hold[hold] = w
        logger.info("hold=%d: %d windows", hold, len(w))

    primary = compare(windows_by_hold[PRIMARY_N], PRIMARY_DECAY, PRIMARY_N)
    cells = [compare(windows_by_hold[n], d, n) for d, n in SENSITIVITY]
    apply_holm(cells)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(primary, cells, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_opening_run_%s.md" % run_id)).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_opening_run_%s.json" % run_id)).write_text(
        json.dumps({"primary": {k: v for k, v in primary.items() if k != "rows"},
                    "sensitivity": [{k: v for k, v in c.items() if k != "rows"}
                                    for c in cells]}, indent=2), encoding="utf-8")
    print(report.encode("ascii", "replace").decode("ascii"))
    return primary


if __name__ == "__main__":
    main()
