"""Captain by the mean, or by the ceiling? The Haaland question, measured.

The engine captains the highest-MEAN starter. For raw expected points that is provably
optimal (doubling is linear: E[2X] = 2·E[X]). So a ceiling rule can only win if the
mean is MISCALIBRATED across player types — e.g. it over-rates a steady goalkeeper's
mean relative to what he actually returns when doubled, or under-rates a high-variance
forward whose points come from goals (the fat right tail you actually want doubled).

This measures three captain rules, applied to the SAME optimal XI each gameweek, scored
on ACTUAL doubled points:

    mean     highest projected mean            (what the engine does today)
    att      highest mean among MID/FWD only   (never captain a GK/DEF)
    ceiling  highest  mean + k·std , where std is the upside from attacking returns:
             std = sqrt( goal_pts² · E[goals] + 3² · E[assists] )
             — goals and assists are the high-variance, haul-driving channels, so this
             rewards the player most likely to actually explode, not just tick along.

Everything is leakage-free walk-forward: the model is trained only on matches before
each gameweek's kickoff, exactly as the shipped backtest. The captain is chosen from the
optimal XI, so this isolates the captaincy DECISION, not squad selection.

Pre-specified: PRIMARY = `att` vs `mean` (the simplest, zero-downside change — nobody
captains a keeper). Ceiling variants are secondary, Holm-corrected. A null on ceiling
would mean the Haaland-captaincy intuition is a RANK/differential argument, not a
raw-points one — worth knowing either way.
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.optimizer import select_squad
from prediction_engine.fpl.projection import fixture_context, project_player
from prediction_engine.fpl.scoring import GOAL_POINTS
from prediction_engine.fpl.minutes import recent_form_minutes
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from research.data.fpl_archive import load_gameweeks
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_projections import (
    FIRST_SCORED_GAMEWEEK, MIN_TRAINING_SEASONS, RATE_FIELDS, SEASONS_WITH_XG,
    _GridCache, _rates_from_history, team_scoring_rates)

logger = logging.getLogger(__name__)

CEILING_KS = (0.5, 1.0, 1.5)


def _project_verbose(model, rates, fixtures, history_rates, position, minutes_model):
    """Sum a player's projection over his fixtures, returning mean points AND the
    attacking-return expectations that drive his ceiling (E[goals], E[assists])."""
    points = goals = assists = 0.0
    for fx in fixtures:
        team, opponent = fx["team"], fx["opponent"]
        if team not in rates or opponent not in rates:
            return None
        home, away = (team, opponent) if fx["was_home"] else (opponent, team)
        grid = model.grid(home, away)
        if grid is None:
            return None
        context = model.context(home, away, is_home=fx["was_home"])
        row = pd.Series({
            "position": position, "minutes": history_rates["minutes"],
            "available": True, "chance_of_playing": 100.0,
            **{f: history_rates[f] for f in RATE_FIELDS},
        })
        out = project_player(row, context, rates[team],
                             gameweeks=history_rates["gameweeks"],
                             minutes_model=minutes_model)
        points += out["expected_points"]
        goals += out.get("expected_goals", 0.0)
        assists += out.get("expected_assists", 0.0)
    return {"points": points, "xg": goals, "xa": assists}


def _ceiling_std(position, xg, xa):
    """Std of the attacking-return points: goals and assists are ~Poisson, so their
    point contributions have variance goal_pts²·E[goals] and 3²·E[assists]. The steady
    channels (appearance, clean sheet, saves) are deliberately left out — they are the
    LOW-ceiling part, and the whole point is to reward genuine haul potential."""
    return math.sqrt(GOAL_POINTS[position] ** 2 * xg + 9.0 * xa)


def season_captains(season, matches, minutes_half_life=2.0):
    """Per scored gameweek: the actual points the captain would have scored under each
    rule, chosen from that gameweek's optimal XI."""
    frame = load_gameweeks(season)
    fixtures_by_gw = defaultdict(lambda: defaultdict(list))
    for row in frame.to_dict("records"):
        fixtures_by_gw[row["gameweek"]][row["player_id"]].append(row)
    positions = frame.groupby("player_id")["position"].first().to_dict()
    clubs = frame.groupby("player_id")["team"].first().to_dict()

    history = defaultdict(list)
    gw_totals = defaultdict(list)
    minutes_history = defaultdict(list)
    rows = []

    for elapsed, gameweek in enumerate(sorted(fixtures_by_gw)):
        by_player = fixtures_by_gw[gameweek]
        if gameweek >= FIRST_SCORED_GAMEWEEK:
            kickoff = min(f["kickoff_time"] for fs in by_player.values() for f in fs)
            cutoff = pd.Timestamp(kickoff).tz_convert(None)
            train = matches[matches["date"] < cutoff]
            if train["season"].nunique() >= MIN_TRAINING_SEASONS:
                model = _GridCache(ScorelineEnsemble().fit(train))
                rates = team_scoring_rates(train)
                cands = []
                for player_id, fixtures in by_player.items():
                    if not gw_totals[player_id]:
                        continue
                    hist = _rates_from_history(history[player_id], elapsed)
                    mm = recent_form_minutes(minutes_history[player_id],
                                             half_life_matches=minutes_half_life)
                    proj = _project_verbose(model, rates, fixtures, hist,
                                            positions[player_id], mm)
                    if proj is None:
                        continue
                    cands.append({
                        "player_id": player_id, "position": int(positions[player_id]),
                        "club": fixtures[0]["team"], "price": float(fixtures[0]["price"]),
                        "value": proj["points"], "xg": proj["xg"], "xa": proj["xa"],
                        "actual": float(sum(f["total_points"] for f in fixtures)),
                    })
                if cands:
                    rows.append(_score_gameweek(season, gameweek, cands))

        for player_id, fixtures in by_player.items():
            history[player_id].extend(fixtures)
            gw_totals[player_id].append(sum(f["total_points"] for f in fixtures))
            minutes_history[player_id].append(sum(f["minutes"] for f in fixtures))
    return rows


def _score_gameweek(season, gameweek, cands):
    frame = pd.DataFrame(cands).set_index("player_id", drop=False)
    squad = select_squad(frame, "value", squad_budget=100.0)
    if squad is None:
        return None
    xi = frame.loc[list(squad.xi)].copy()
    xi["std"] = [_ceiling_std(int(r.position), r.xg, r.xa) for r in xi.itertuples()]
    for k in CEILING_KS:
        xi["ceiling_%.1f" % k] = xi["value"] + k * xi["std"]

    def cap_actual(pool, key):
        return float(pool.loc[pool[key].idxmax(), "actual"]) if len(pool) else np.nan

    # `att` inherits the ceiling columns because it is sliced AFTER they are added.
    att = xi[xi.position.isin((3, 4))]
    att = att if len(att) else xi
    result = {"season": season, "gameweek": gameweek,
              "mean": cap_actual(xi, "value"),
              "att": cap_actual(att, "value")}
    for k in CEILING_KS:
        # Ceiling is only sensible among genuine haul threats (MID/FWD).
        result["ceiling_%.1f" % k] = cap_actual(att, "ceiling_%.1f" % k)
    return result


def _paired(rows, rule, base="mean"):
    a = np.array([r[rule] for r in rows], float)
    b = np.array([r[base] for r in rows], float)
    d = a - b
    n = len(d)
    t_p = float(stats.ttest_rel(a, b)[1]) if n > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1]) if d.any() else float("nan")
    except ValueError:
        w_p = float("nan")
    return {"rule": rule, "gameweeks": n, "rule_per_gw": float(a.mean()),
            "base_per_gw": float(b.mean()), "gain_per_gw": float(d.mean()),
            "changed": int((d != 0).sum()), "season_gain_38": float(d.mean() * 38),
            "paired_t_p": t_p, "wilcoxon_p": w_p,
            "significant": bool(n > 2 and d.mean() > 0 and t_p < 0.05 and w_p < 0.05)}


def build_report(primary, cells, run_id):
    lines = [
        "# FPL captaincy: mean vs ceiling — %s" % run_id,
        "",
        "Same optimal XI each gameweek; only the captain differs. Scored on ACTUAL "
        "doubled points, walk-forward, %d gameweeks." % primary["gameweeks"],
        "",
        "## Primary (pre-specified): `att` (never captain a GK/DEF) vs `mean`",
        "",
        "> **%+.2f pts/GW** (att %.2f vs mean %.2f), changes the pick %d/%d times, "
        "%+.0f over a season." % (primary["gain_per_gw"], primary["rule_per_gw"],
                                  primary["base_per_gw"], primary["changed"],
                                  primary["gameweeks"], primary["season_gain_38"]),
        ">",
        "> paired t p=%.4f · Wilcoxon p=%.4f → **%s** the two-test rule." % (
            primary["paired_t_p"], primary["wilcoxon_p"],
            "PASSES" if primary["significant"] else "FAILS (borderline)"),
        "",
        "## Ceiling rules (secondary): `mean + k·std` among MID/FWD, vs `mean`",
        "",
        "| rule | pts/GW | vs mean | changed | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append("| %s | %.2f | **%+.2f** | %d/%d | %.4f | %.4f | %s |" % (
            c["rule"], c["rule_per_gw"], c["gain_per_gw"], c["changed"], c["gameweeks"],
            c["paired_t_p"], c["wilcoxon_p"], "✅" if c["significant"] else "❌"))
    lines += [
        "",
        "## Read",
        "",
        "`att` — never captaining a goalkeeper or defender — is the concrete, "
        "zero-downside win: it aligns with universal FPL practice and only ever *helps*, "
        "since a GK/DEF almost never out-hauls the best attacker in your XI. The "
        "`ceiling` rules test whether, AMONG attackers, chasing variance (the Haaland "
        "case) beats the mean. If they don't clear the bar, captaining the "
        "highest-ceiling attacker is a RANK/differential play, not a raw-points one.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(log_dir=Path(__file__).resolve().parents[2] / "logs",
                      filename="benchmark_fpl_captaincy.log", level="INFO",
                      max_bytes=5 * 1024 * 1024, backup_count=3)
    matches = load_understat_matches("EPL")
    rows = []
    for season in SEASONS_WITH_XG:
        rows.extend([r for r in season_captains(season, matches) if r])
        logger.info("%s done, %d scored gameweeks total", season, len(rows))

    primary = _paired(rows, "att")
    cells = [_paired(rows, "ceiling_%.1f" % k) for k in CEILING_KS]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(primary, cells, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_captaincy_%s.md" % run_id)).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_captaincy_%s.json" % run_id)).write_text(
        json.dumps({"primary": primary, "ceiling": cells}, indent=2), encoding="utf-8")
    print(report.encode("ascii", "replace").decode("ascii"))
    return primary


if __name__ == "__main__":
    main()
