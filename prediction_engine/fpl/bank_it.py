"""Bank It: the engine's optimal £100m squad for one gameweek, locked before the deadline.

This is the live forward-validation pipeline (see docs/04_BANK_IT_PIPELINE.md). The
edge is proven in backtest; the only honest way to know it holds forward is to
commit a squad BEFORE a gameweek's deadline and score it after. This module produces
that squad.

It is deliberately thin: every hard part already exists and is validated.

    project_fixture   the shipped per-fixture projection (ensemble grid + player
                      rates + recent-form minutes + live injury flags)
    select_squad      the provably-optimal 15-man squad solver (branch and bound,
                      verified against exhaustive enumeration)

The one genuinely new job here is ASSEMBLY: `project_fixture` projects a single
fixture, but the optimizer needs every player across every fixture in the gameweek
in one frame. `build_gameweek_frame` runs the projection over each fixture and
stitches the results into the optimizer's contract, handling the two cases a single
fixture never sees:

    double gameweeks  a player with two fixtures becomes ONE row whose points are
                      summed (else he could be picked twice);
    blank gameweeks   a player with no fixture is simply absent, so unpickable.

What it deliberately does NOT do (v1): carry a squad or model transfers. It rebuilds
from scratch each gameweek, exactly as the backtest that proved the edge does, so the
live number is comparable to the proven one. Transfers are a later layer.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from prediction_engine.engine import PredictionEngine
from prediction_engine.fpl.cli import _recent_minutes_history, _safe
from prediction_engine.fpl.optimizer import TOTAL_SQUAD_BUDGET, select_squad
from prediction_engine.fpl.projection import project_fixture
from research.data.fpl_loader import (
    fixtures_for_gameweek,
    load_players,
    next_gameweek,
)
from research.data.lineup_resolver import resolve_snapshot
from research.data.predicted_lineups import ARCHIVE_DIR, current_season

logger = logging.getLogger(__name__)

# Point channels are additive across a player's fixtures in a gameweek: a double
# gameweek really is two lots of points. Identity and rate columns are not summed.
_ADDITIVE = (
    "expected_points", "expected_minutes", "appearance", "goals", "assists",
    "clean_sheet", "conceded", "saves", "bonus", "defensive", "cards",
    "expected_goals", "expected_assists",
)
_FIRST = ("player", "team", "position", "position_id", "price", "available",
          "appearance_factor", "clean_sheet_probability")

Projector = Callable[..., pd.DataFrame]


def latest_snapshot_before(deadline_time: Optional[str], season: str,
                           archive_dir=ARCHIVE_DIR):
    """The newest archived predicted-lineup snapshot fetched BEFORE the deadline.

    The filter is the whole point. A snapshot fetched after the deadline knows things
    we could not have known when the squad was locked — team news, late fitness tests,
    even the confirmed XI — so reading one would quietly turn the forward test into
    hindsight and invalidate the very claim Bank-It exists to make. Timestamps are
    compared as ISO-8601 Z strings, which both sides emit in the same fixed format, so
    lexicographic order is chronological order.

    Returns (snapshot, path) or None.
    """
    folder = Path(archive_dir) / season
    if not folder.exists():
        return None

    best, best_path = None, None
    for path in sorted(folder.glob("*.json")):
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("unreadable snapshot %s — skipping", path)
            continue
        fetched = snapshot.get("fetched_at")
        if not fetched:
            continue
        if deadline_time and fetched >= deadline_time:
            continue                       # published too late to be honest evidence
        if best is None or fetched > best["fetched_at"]:
            best, best_path = snapshot, path
    return None if best is None else (best, best_path)


def lineup_start_pct(players: pd.DataFrame, season: str,
                     deadline_time: Optional[str], archive_dir=ARCHIVE_DIR):
    """({player_id: Start %}, provenance) from the newest pre-deadline snapshot.

    Returns ({}, None) when nothing usable is archived, so the caller simply keeps the
    recent-form model rather than failing: a missing feed must degrade the projection,
    never block the squad.
    """
    found = latest_snapshot_before(deadline_time, season, archive_dir)
    if found is None:
        logger.warning("no pre-deadline lineup snapshot for %s — using recent form", season)
        return {}, None

    snapshot, path = found
    resolved = resolve_snapshot(snapshot, players)
    stats = resolved["stats"]
    provenance = {
        "snapshot": Path(path).name,
        "source": snapshot.get("source"),
        "fetched_at": snapshot.get("fetched_at"),
        "teams": stats["teams"],
        "matched": stats["matched"],
        "unmatched": stats["unmatched"],
        "match_rate": stats["match_rate"],
    }
    logger.info("lineups: %s (%d matched, %d unmatched, %.1f%%)",
                provenance["snapshot"], stats["matched"], stats["unmatched"],
                100 * stats["match_rate"])
    return resolved["start_pct"], provenance


def build_gameweek_frame(engine, players: pd.DataFrame,
                         fixtures: List[Tuple[str, str]],
                         minutes_history: Optional[Dict[int, list]] = None,
                         start_pct: Optional[Dict[int, float]] = None,
                         projector: Projector = project_fixture) -> pd.DataFrame:
    """Every player's expected points for one gameweek, as the optimizer wants it.

    Projects each fixture with `projector` (the shipped `project_fixture` by
    default; injectable so the assembly is testable without the full engine), then:
      - sums each player's point channels across his fixtures (double gameweeks);
      - collapses him to a single row;
      - maps `position` to the integer 1-4 the optimizer expects and adds the
        `club` column it keys the 3-per-club rule on.

    A player absent from every fixture (a blank gameweek) never appears, so he is
    correctly unpickable. Raises ValueError if `fixtures` is empty.
    """
    if not fixtures:
        raise ValueError("no fixtures to project for this gameweek")

    tables = [projector(engine, players, home, away, minutes_history=minutes_history,
                        start_pct=start_pct)
              for home, away in fixtures]
    allrows = pd.concat(tables, ignore_index=True)

    agg = {channel: "sum" for channel in _ADDITIVE if channel in allrows.columns}
    for column in _FIRST:
        if column in allrows.columns:
            agg[column] = "first"
    if "opponent" in allrows.columns:                # join both opponents in a DGW
        agg["opponent"] = lambda values: ", ".join(dict.fromkeys(values))

    frame = allrows.groupby("player_id", as_index=False).agg(agg)

    # The optimizer reads integer `position` and a `club` column; the projection
    # emits a display string and `team`. Keep the string for humans as
    # `position_name`, and expose the integer as `position`.
    frame = frame.rename(columns={"position": "position_name"})
    frame["position"] = frame["position_id"].astype(int)
    frame["club"] = frame["team"]
    return frame


def pick_squad(frame: pd.DataFrame, budget: float = TOTAL_SQUAD_BUDGET):
    """The provably-optimal legal £`budget`m squad for this gameweek's projections.

    Returns a `SquadSelection` (xi, bench, projected, cost, formation, captain), or
    None if no legal squad exists (e.g. a gameweek too blank to field one)."""
    return select_squad(frame, "expected_points", squad_budget=budget)


def baseline_ppg(players: pd.DataFrame, minutes_history: Optional[Dict[int, list]],
                 prior_ppg: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """Season-to-date points-per-gameweek per player id — the pre-registered baseline.

    ppg = total points / gameweeks the player actually had a fixture (the same
    denominator the backtest's `player_ppg` uses). A player with no fixture yet
    (opening weeks, new signings) falls back to `prior_ppg` — last season's ppg,
    supplied at pre-registration — else 0. Uses only pre-deadline information, so
    the baseline squad can be locked before the deadline exactly as ours is.
    """
    prior_ppg = prior_ppg or {}
    minutes_history = minutes_history or {}
    ppg = {}
    for player_id, points in zip(players["id"].astype(int), players["total_points"].astype(float)):
        played = len(minutes_history.get(player_id, []))
        ppg[player_id] = (points / played) if played > 0 else float(prior_ppg.get(player_id, 0.0))
    return ppg


# ---------------------------------------------------------------------------
# Artifact: the committed record of what we predicted, before the deadline.
# ---------------------------------------------------------------------------

def _row_record(row: pd.Series, is_captain: bool = False,
                is_vice: bool = False) -> dict:
    return {
        "player_id": int(row["player_id"]),
        "player": str(row["player"]),
        "team": str(row["team"]),
        "position": str(row["position_name"]),
        "price": round(float(row["price"]), 1),
        "expected_points": round(float(row["expected_points"]), 3),
        "available": bool(row["available"]),
        "captain": bool(is_captain),
        "vice_captain": bool(is_vice),
    }


def _vice_of(frame: pd.DataFrame, squad, value_column: str):
    """The vice-captain: the highest-`value_column` starter who is not the captain.
    FPL promotes him to captain if the captain plays 0 minutes."""
    ranked = frame.loc[squad.xi].sort_values(value_column, ascending=False)
    for idx in ranked.index:
        if idx != squad.captain:
            return idx
    return squad.captain              # 11-man XI always has a distinct second; guard only


def _baseline_block(frame: pd.DataFrame, baseline_squad) -> dict:
    """A compact record of the locked baseline squad — player ids are enough to
    score it after the gameweek, alongside the human-readable XI."""
    defenders, midfielders, forwards = baseline_squad.formation
    xi = frame.loc[baseline_squad.xi].sort_values("player_ppg", ascending=False)
    vice_idx = _vice_of(frame, baseline_squad, "player_ppg")
    return {
        "name": "player_ppg",
        "formation": "%d-%d-%d" % (defenders, midfielders, forwards),
        "squad_cost": round(float(baseline_squad.cost), 1),
        "captain_id": int(frame.loc[baseline_squad.captain, "player_id"]),
        "vice_captain_id": int(frame.loc[vice_idx, "player_id"]),
        "xi": [int(frame.loc[idx, "player_id"]) for idx in baseline_squad.xi],
        "bench": [int(frame.loc[idx, "player_id"]) for idx in baseline_squad.bench],
        "xi_names": [str(frame.loc[idx, "player"]) for _, idx in
                     zip(range(len(xi)), xi.index)],
    }


def build_artifact(frame: pd.DataFrame, squad, gameweek: int,
                   deadline: Optional[str], config: dict, baseline_squad=None) -> dict:
    """The full, JSON-serialisable record of this gameweek's prediction.

    When `baseline_squad` is supplied it is locked into the artifact too, so the
    pre-registered paired comparison (ours vs `player_ppg`) is fixed before the
    deadline — neither squad can be reconstructed with hindsight at scoring time.
    """
    xi = frame.loc[squad.xi].sort_values(["position_id", "expected_points"],
                                         ascending=[True, False])
    bench = frame.loc[squad.bench].sort_values(["position_id", "expected_points"],
                                               ascending=[True, False])
    vice_idx = _vice_of(frame, squad, "expected_points")
    defenders, midfielders, forwards = squad.formation
    artifact = {
        "gameweek": int(gameweek),
        "deadline_time": deadline,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": config,
        "formation": "%d-%d-%d" % (defenders, midfielders, forwards),
        "budget": round(float(config.get("budget", TOTAL_SQUAD_BUDGET)), 1),
        "squad_cost": round(float(squad.cost), 1),
        "projected_points": round(float(squad.projected), 3),
        "captain_id": int(frame.loc[squad.captain, "player_id"]),
        "vice_captain_id": int(frame.loc[vice_idx, "player_id"]),
        "xi": [_row_record(row, is_captain=(idx == squad.captain),
                           is_vice=(idx == vice_idx))
               for idx, row in xi.iterrows()],
        "bench": [_row_record(row) for _, row in bench.iterrows()],
    }
    if baseline_squad is not None:
        artifact["baseline"] = _baseline_block(frame, baseline_squad)
    return artifact


def render_markdown(artifact: dict) -> str:
    lines = [
        "# Bank It — GW%d squad" % artifact["gameweek"],
        "",
        "*Generated %s%s. Formation %s · cost £%.1fm · projected %.1f pts.*" % (
            artifact["generated_at"],
            (" · deadline %s" % artifact["deadline_time"]) if artifact["deadline_time"] else "",
            artifact["formation"], artifact["squad_cost"], artifact["projected_points"]),
        "",
        "## Starting XI",
        "",
        "| pos | player | team | £ | xPts | |",
        "|---|---|---|---|---|---|",
    ]
    for r in artifact["xi"]:
        mark = "(C)" if r["captain"] else ("(V)" if r.get("vice_captain") else "")
        lines.append("| %s | %s | %s | %.1f | %.2f | %s |" % (
            r["position"], r["player"], r["team"], r["price"], r["expected_points"], mark))
    lines += ["", "## Bench", "", "| pos | player | team | £ | xPts |", "|---|---|---|---|---|"]
    for r in artifact["bench"]:
        lines.append("| %s | %s | %s | %.1f | %.2f |" % (
            r["position"], r["player"], r["team"], r["price"], r["expected_points"]))
    return "\n".join(lines)


def bank_gameweek(gameweek: Optional[int] = None, budget: float = TOTAL_SQUAD_BUDGET,
                  prior_ppg: Optional[Dict[int, float]] = None,
                  use_lineups: bool = False) -> dict:
    """Load live data, project the target gameweek, and return the squad artifact.

    Locks BOTH our squad and the `player_ppg` baseline squad (Section 3 of the
    design doc), so the pre-registered comparison is fixed before the deadline.
    `prior_ppg` supplies last-season ppg for the opening weeks before a current-
    season average exists.

    `gameweek` None auto-detects the upcoming gameweek; off-season (no upcoming
    gameweek published) that raises, because there is nothing to bank yet.
    """
    engine = PredictionEngine.train("EPL")
    players = load_players()
    minutes_history = _recent_minutes_history()

    deadline = None
    if gameweek is None:
        upcoming = next_gameweek()
        if upcoming is None:
            raise ValueError("no upcoming gameweek is published (off-season) — "
                             "pass --gameweek to dry-run a specific one")
        gameweek, deadline = upcoming["gameweek"], upcoming["deadline_time"]

    fixtures = fixtures_for_gameweek(gameweek)
    if not fixtures:
        raise ValueError("no fixtures scheduled for gameweek %d" % gameweek)

    # The Phase 6f candidate, opt-in. Only ever reads a snapshot archived BEFORE this
    # gameweek's deadline (see `latest_snapshot_before`).
    start_pct, lineup_provenance = ({}, None)
    if use_lineups:
        start_pct, lineup_provenance = lineup_start_pct(
            players, current_season(), deadline)

    frame = build_gameweek_frame(engine, players, fixtures,
                                 minutes_history=minutes_history,
                                 start_pct=start_pct or None)
    squad = pick_squad(frame, budget=budget)
    if squad is None:
        raise ValueError("no legal squad exists for gameweek %d" % gameweek)

    # The baseline squad, locked from the same frame with only pre-deadline info.
    frame["player_ppg"] = frame["player_id"].map(
        baseline_ppg(players, minutes_history, prior_ppg)).fillna(0.0)
    baseline_squad = select_squad(frame, "player_ppg", squad_budget=budget)

    minutes_model = "recent-form (half-life 2)" if minutes_history else "crude flat-average"
    if start_pct:
        minutes_model += " + predicted-lineup Start %"
    config = {
        "budget": budget,
        "xg_source": "fpl-opta (live bootstrap)",
        "minutes_model": minutes_model,
        "engine": "ScorelineEnsemble (Elo + Poisson-xG + Dixon-Coles-xG)",
        "fixtures": len(fixtures),
        # Which lineup snapshot fed this squad, recorded in the locked artifact so the
        # committed prediction says exactly what it knew and when it knew it.
        "lineups": lineup_provenance or "off",
    }
    return build_artifact(frame, squad, gameweek, deadline, config,
                          baseline_squad=baseline_squad)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prediction_engine.fpl.bank_it",
        description="The engine's optimal £100m FPL squad for a gameweek, locked "
                    "before the deadline.")
    parser.add_argument("--gameweek", type=int, default=None,
                        help="target gameweek (default: the next one; off-season, required)")
    parser.add_argument("--budget", type=float, default=TOTAL_SQUAD_BUDGET,
                        help="squad budget in £m (default 100.0)")
    parser.add_argument("--out", type=str, default=None,
                        help="directory to write GWxx.{json,md} (default: print only)")
    parser.add_argument("--lineups", action="store_true",
                        help="use the archived pre-deadline predicted-lineup Start %% "
                             "as the minutes signal (Phase 6f candidate; UNPROVEN, "
                             "and only ever reads a snapshot archived before the "
                             "deadline)")
    return parser


def main(argv: List[str] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = _build_parser().parse_args(argv)

    try:
        artifact = bank_gameweek(gameweek=args.gameweek, budget=args.budget,
                                 use_lineups=args.lineups)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    markdown = render_markdown(artifact)
    print(_safe(markdown))

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "GW%02d" % artifact["gameweek"]
        (out_dir / (stem + ".json")).write_text(json.dumps(artifact, indent=2),
                                                 encoding="utf-8")
        (out_dir / (stem + ".md")).write_text(markdown, encoding="utf-8")
        print("\nwrote %s.{json,md} to %s" % (stem, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
