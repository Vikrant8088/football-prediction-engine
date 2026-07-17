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


# docs/05 pre-registers this: the opening weeks use LAST season's ppg, because a
# current-season average built on one or two games is noise — and at GW1 it does not
# exist at all, which would leave the baseline squad degenerate (every player tied on
# zero) and the comparison meaningless.
EARLY_SEASON_GAMEWEEKS = 5


def previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(season.split("-")[0])
    return "%d-%s" % (start - 1, str(start)[-2:])


def last_season_ppg(players: pd.DataFrame, season: str = None) -> Dict[int, float]:
    """{current player id: last season's points-per-gameweek}.

    Joined on FPL's permanent `code`, never on the element id: ids are reassigned
    every summer, so an id join would silently credit one player with another's
    season. Players with no last season (new signings, promoted clubs) are simply
    absent — the caller treats them as 0 rather than inventing a history.
    """
    from research.data.fpl_archive import ALL_SEASONS, load_gameweeks, load_player_meta

    if season is None:
        # Deliberately NOT ALL_SEASONS[-1]: the moment the current season is added to
        # that list, it would return the season we are playing rather than the one
        # before it, quietly leaking live points into the "prior" baseline.
        season = previous_season(current_season())
        if season not in ALL_SEASONS:
            season = ALL_SEASONS[-1]
            logger.warning("no archive for the previous season; falling back to %s", season)
    frame = load_gameweeks(season)
    # Points per gameweek he actually had a fixture — the same denominator the
    # backtest's player_ppg uses. Double gameweeks sum into their one gameweek.
    per_gameweek = frame.groupby(["player_id", "gameweek"])["total_points"].sum()
    totals = per_gameweek.groupby(level=0).agg(["sum", "count"])

    meta = load_player_meta(season)
    by_code = {}
    for player_id, row in totals.iterrows():
        info = meta.get(int(player_id)) or {}
        code, games = info.get("code"), int(row["count"])
        if code is None or games <= 0:
            continue
        by_code[int(code)] = float(row["sum"]) / games

    prior = {}
    for player_id, code in zip(players["id"].astype(int), players["code"].astype(int)):
        ppg = by_code.get(int(code))
        if ppg is not None:
            prior[int(player_id)] = ppg
    logger.info("last-season ppg (%s): %d/%d current players matched by code",
                season, len(prior), len(players))
    return prior


# The per-90 rates the projection consumes. All are season-to-date in FPL's bootstrap,
# which means they are all ZERO on the opening day of a season.
RATE_FIELDS = ("xg_per_90", "xa_per_90", "saves_per_90", "bonus_per_90",
               "dc_per_90", "cards_per_90")


def last_season_rates(players: pd.DataFrame, season: str = None) -> Dict[int, dict]:
    """{current player id: last season's per-90 rates}, joined on the permanent `code`.

    Why this must exist: FPL resets every season-to-date stat to zero, so on the
    opening day `expected_goals` is 0.00 for everyone and `xg_per_90` collapses to 0.
    Without this, a GW1 squad is picked on appearance points and clean sheets alone —
    Haaland projects the same as any other nailed-on striker. The rates are computed
    exactly as the backtest's `_rates_from_history` does (totals over minutes/90), so
    the cold-start speaks the same language as the validated model.

    Honest limits: a player with no previous PREMIER LEAGUE season is simply absent —
    a summer signing from abroad has no history here and stays invisible until he
    plays. And last season is not current form: it cannot see transfers, the World
    Cup, or pre-season.
    """
    from research.data.fpl_archive import ALL_SEASONS, load_gameweeks, load_player_meta

    if season is None:
        season = previous_season(current_season())
        if season not in ALL_SEASONS:
            season = ALL_SEASONS[-1]
            logger.warning("no archive for the previous season; falling back to %s", season)

    frame = load_gameweeks(season)
    totals = frame.groupby("player_id").agg({
        "minutes": "sum", "expected_goals": "sum", "expected_assists": "sum",
        "saves": "sum", "bonus": "sum", "yellow_cards": "sum", "red_cards": "sum",
        **({"defensive_contribution": "sum"}
           if "defensive_contribution" in frame.columns else {}),
    })

    meta = load_player_meta(season)
    by_code = {}
    for player_id, row in totals.iterrows():
        info = meta.get(int(player_id)) or {}
        code = info.get("code")
        minutes = float(row["minutes"])
        if code is None or minutes <= 0:
            continue
        per_90 = minutes / 90.0
        by_code[int(code)] = {
            "xg_per_90": float(row["expected_goals"]) / per_90,
            "xa_per_90": float(row["expected_assists"]) / per_90,
            "saves_per_90": float(row["saves"]) / per_90,
            "bonus_per_90": float(row["bonus"]) / per_90,
            "dc_per_90": (float(row["defensive_contribution"]) / per_90
                          if "defensive_contribution" in totals.columns else 0.0),
            "cards_per_90": (float(row["yellow_cards"]) + 3.0 * float(row["red_cards"])) / per_90,
        }

    rates = {}
    for player_id, code in zip(players["id"].astype(int), players["code"].astype(int)):
        prior = by_code.get(int(code))
        if prior is not None:
            rates[int(player_id)] = prior
    logger.info("last-season rates (%s): %d/%d current players matched by code",
                season, len(rates), len(players))
    return rates


def apply_early_season_rates(players: pd.DataFrame,
                             prior_rates: Dict[int, dict]) -> pd.DataFrame:
    """Substitute last season's per-90 rates for the opening weeks, returning a copy.

    Only fills rates that are absent this season (a player with real current-season
    minutes keeps his own numbers), so this is a cold-start, not an override.
    """
    if not prior_rates:
        return players
    patched = players.copy()
    # Force float columns before writing. If a rate column ever arrives as int dtype,
    # pandas truncates 0.777 to 0 on assignment WITHOUT complaining — which would zero
    # the cold-start silently and hand us back the very GW1 bug this function exists to
    # fix, while looking like it worked.
    for field in RATE_FIELDS:
        if field in patched.columns:
            patched[field] = patched[field].astype(float)
    filled = 0
    for index, row in patched.iterrows():
        prior = prior_rates.get(int(row["id"]))
        if prior is None or float(row["minutes"]) > 0:
            continue                      # he has played this season: trust the live data
        for field in RATE_FIELDS:
            patched.at[index, field] = prior[field]
        filled += 1
    logger.info("cold-start rates applied to %d players with no minutes this season", filled)
    return patched


def baseline_ppg(players: pd.DataFrame, minutes_history: Optional[Dict[int, list]],
                 prior_ppg: Optional[Dict[int, float]] = None,
                 gameweek: Optional[int] = None) -> Dict[int, float]:
    """Points-per-gameweek per player id — the pre-registered `player_ppg` baseline.

    ppg = total points / gameweeks the player actually had a fixture (the same
    denominator the backtest uses). Per docs/05, gameweeks 1-%d instead use LAST
    season's ppg, since a one- or two-game average is noise and at GW1 there is no
    average at all. From GW%d it switches to the current season, falling back to last
    season's for a player with no history yet. A player with neither scores 0, which
    correctly makes him unpickable by the baseline rather than falsely attractive.

    Uses only pre-deadline information, so the baseline squad locks exactly as ours does.
    """ % (EARLY_SEASON_GAMEWEEKS, EARLY_SEASON_GAMEWEEKS + 1)
    prior_ppg = prior_ppg or {}
    minutes_history = minutes_history or {}
    early = gameweek is not None and gameweek <= EARLY_SEASON_GAMEWEEKS

    ppg = {}
    for player_id, points in zip(players["id"].astype(int), players["total_points"].astype(float)):
        played = len(minutes_history.get(player_id, []))
        if early or played == 0:
            ppg[player_id] = float(prior_ppg.get(player_id, 0.0))
        else:
            ppg[player_id] = points / played
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


def _squad_block(frame: pd.DataFrame, squad, value_column: str, name: str,
                 extra: dict = None) -> dict:
    """A compact record of a locked squad — player ids are enough to score it after
    the gameweek, alongside a human-readable XI. Used for the baseline and for any
    declared variant."""
    defenders, midfielders, forwards = squad.formation
    ranked = frame.loc[squad.xi].sort_values(value_column, ascending=False)
    vice_idx = _vice_of(frame, squad, value_column)
    block = {
        "name": name,
        "formation": "%d-%d-%d" % (defenders, midfielders, forwards),
        "squad_cost": round(float(squad.cost), 1),
        "projected_points": round(float(squad.projected), 3),
        "captain_id": int(frame.loc[squad.captain, "player_id"]),
        "vice_captain_id": int(frame.loc[vice_idx, "player_id"]),
        "xi": [int(frame.loc[idx, "player_id"]) for idx in squad.xi],
        "bench": [int(frame.loc[idx, "player_id"]) for idx in squad.bench],
        "xi_names": [str(frame.loc[idx, "player"]) for idx in ranked.index],
    }
    if extra:
        block.update(extra)
    return block


def _baseline_block(frame: pd.DataFrame, baseline_squad) -> dict:
    return _squad_block(frame, baseline_squad, "player_ppg", "player_ppg")


def build_artifact(frame: pd.DataFrame, squad, gameweek: int,
                   deadline: Optional[str], config: dict, baseline_squad=None,
                   variants: dict = None) -> dict:
    """The full, JSON-serialisable record of this gameweek's prediction.

    When `baseline_squad` is supplied it is locked into the artifact too, so the
    pre-registered paired comparison (ours vs `player_ppg`) is fixed before the
    deadline — neither squad can be reconstructed with hindsight at scoring time.

    `variants` locks additional, separately-declared squads as
    {name: (squad, value_column, extra)}. The top-level squad is always the
    pre-registered PRIMARY (the proven recent-form champion); a variant is a candidate
    under forward test, never the headline. Locking both in one artifact is what makes
    the comparison paired — same gameweek, same prices, same deadline.
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
    if variants:
        artifact["variants"] = {
            name: _squad_block(frame, variant_squad, value_column, name, extra)
            for name, (variant_squad, value_column, extra) in variants.items()
        }
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
                  use_lineups: bool = True) -> dict:
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

    # Opening weeks: FPL has zeroed every season-to-date stat, so without last
    # season's rates every xg_per_90 is 0 and the squad is picked on appearance
    # points alone. Cold-start those rates for players yet to feature.
    early = bool(gameweek and gameweek <= EARLY_SEASON_GAMEWEEKS)
    cold_started = 0
    if early:
        try:
            prior_rates = last_season_rates(players)
            before = players
            players = apply_early_season_rates(players, prior_rates)
            cold_started = int((before["minutes"] == 0).sum()) if prior_rates else 0
        except Exception as exc:            # never block the squad on a missing archive
            logger.warning("no last-season rates (%s); opening-week projections will be "
                           "weak", exc)

    # The PRIMARY squad is always the proven recent-form champion, with no lineup
    # feed: docs/05 pre-registers it, and the backtest validated it. The Start %
    # variant below is a candidate under forward test, never the headline.
    frame = build_gameweek_frame(engine, players, fixtures,
                                 minutes_history=minutes_history)
    squad = pick_squad(frame, budget=budget)
    if squad is None:
        raise ValueError("no legal squad exists for gameweek %d" % gameweek)

    # The Phase 6f candidate, locked ALONGSIDE the primary rather than replacing it —
    # that is what makes it a clean paired A/B (same gameweek, prices and deadline)
    # instead of a confounded switch. Only ever reads a snapshot archived BEFORE this
    # gameweek's deadline (see `latest_snapshot_before`).
    variants = {}
    start_pct, lineup_provenance = ({}, None)
    if use_lineups:
        start_pct, lineup_provenance = lineup_start_pct(
            players, current_season(), deadline)
    if start_pct:
        lineup_frame = build_gameweek_frame(engine, players, fixtures,
                                            minutes_history=minutes_history,
                                            start_pct=start_pct)
        # Same fixtures and squad -> same rows, so the variant's projection joins onto
        # the primary frame and every squad indexes the identical player set.
        frame["expected_points_lineups"] = frame["player_id"].map(
            dict(zip(lineup_frame["player_id"], lineup_frame["expected_points"])))
        lineup_squad = select_squad(frame, "expected_points_lineups", squad_budget=budget)
        if lineup_squad is not None:
            variants["lineups"] = (lineup_squad, "expected_points_lineups",
                                   {"minutes_model": "recent-form + predicted-lineup "
                                                     "Start %",
                                    "status": "UNPROVEN candidate under forward test "
                                              "(Phase 6f)",
                                    "lineups": lineup_provenance})

    # The baseline squad, locked from the same frame with only pre-deadline info.
    # prior_ppg is computed, not merely accepted: docs/05 pre-registers last season's
    # ppg for the opening weeks, and without it GW1's baseline would be every player
    # tied on zero — a degenerate squad, and a meaningless comparison.
    if prior_ppg is None:
        try:
            prior_ppg = last_season_ppg(players)
        except Exception as exc:            # a missing archive must not block the squad
            logger.warning("no last-season ppg (%s); baseline falls back to "
                           "current-season only", exc)
            prior_ppg = {}
    frame["player_ppg"] = frame["player_id"].map(
        baseline_ppg(players, minutes_history, prior_ppg, gameweek=gameweek)).fillna(0.0)
    baseline_squad = select_squad(frame, "player_ppg", squad_budget=budget)

    config = {
        "budget": budget,
        "xg_source": "fpl-opta (live bootstrap)",
        # The primary's minutes model. The variant records its own.
        "minutes_model": ("recent-form (half-life 2)" if minutes_history
                          else "crude flat-average"),
        "engine": "ScorelineEnsemble (Elo + Poisson-xG + Dixon-Coles-xG)",
        "fixtures": len(fixtures),
        "early_season_baseline": early,
        # GW1-5 sit OUTSIDE the range the backtest ever scored (it starts at GW6), and
        # their rates are cold-started from last season. Flagged in the locked artifact
        # so an exploratory gameweek can never be mistaken for a validated one.
        "cold_started_rates": cold_started,
        "scoring_status": ("EXPLORATORY — GW1-%d is outside the backtest's validated "
                           "range (GW6+) and rates are cold-started from last season"
                           % EARLY_SEASON_GAMEWEEKS) if early else "primary (validated range)",
        # Which lineup snapshot fed the variant, recorded in the locked artifact so
        # the committed prediction says exactly what it knew and when it knew it.
        "lineups": lineup_provenance or "off",
    }
    return build_artifact(frame, squad, gameweek, deadline, config,
                          baseline_squad=baseline_squad, variants=variants)


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
    parser.add_argument("--no-lineups", dest="lineups", action="store_false",
                        help="skip the predicted-lineup variant. By default it is "
                             "locked ALONGSIDE the primary squad (never replacing it) "
                             "so the Phase 6f Start %% signal gets a clean paired "
                             "forward test; only snapshots archived before the "
                             "deadline are ever read")
    parser.set_defaults(lineups=True)
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
