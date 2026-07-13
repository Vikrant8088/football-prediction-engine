"""Replace FPL's per-match xG with Understat's, so one model spans every season.

The FPL projection is built on per-player xG. FPL only published it from 2022/23,
so to backtest earlier seasons *with the same model* the xG must come from Understat
(which measured it back to 2014/15). This module does the substitution: given a
season's FPL gameweek frame, it overwrites `expected_goals` / `expected_assists`
row by row with the matching Understat figures.

The match is by (player, date):

    FPL element  --build_map-->  Understat id      (per season, name + club)
    Understat id + fixture date  --->  that player's Understat match that day

A fixture with no Understat counterpart (an unmatched player, or a player Understat
has no shot data for that day) keeps xG = 0 and is counted as uncovered. The join is
graded, never assumed: `inject_understat_xg` returns a coverage summary alongside the
frame, and the caller refuses a season whose xG-minutes coverage is too thin.

Applying this to 2022/23-2024/25 - where FPL's OWN xG also exists - is the
validation: if the Understat-fed backtest reproduces the FPL-fed one, the pipeline
is trustworthy on the earlier seasons where only Understat has the data.
"""

import logging
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd

from research.data.understat_fpl_player_map import build_map
from research.data.understat_player_matches import ensure_player_matches

logger = logging.getLogger(__name__)

# A fixture and its Understat match may be dated a day apart (late kickoff, timezone).
DATE_TOLERANCE_DAYS = 1


def _season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def _naive_day(value) -> pd.Timestamp:
    """A timezone-naive calendar day. FPL kickoffs are UTC-aware, Understat dates are
    naive; they cannot be compared until both are stripped to a bare date."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def _matches_by_date(rows: list, start_year: int) -> Dict[pd.Timestamp, dict]:
    """One Understat player's matches in the given season, keyed by calendar day."""
    by_day = {}
    for match in rows:
        if int(match.get("season", -1)) != start_year:
            continue
        by_day[_naive_day(match["date"])] = match
    return by_day


def _lookup(by_day: Dict[pd.Timestamp, dict], day: pd.Timestamp):
    """The Understat match on `day`, or within a day of it (nearest wins)."""
    if day in by_day:
        return by_day[day]
    best, best_gap = None, None
    for other, match in by_day.items():
        gap = abs((other - day).days)
        if gap <= DATE_TOLERANCE_DAYS and (best_gap is None or gap < best_gap):
            best, best_gap = match, gap
    return best


def inject_understat_xg(frame: pd.DataFrame, season: str,
                        matches: Dict[str, list] = None) -> Tuple[pd.DataFrame, dict]:
    """Return `frame` with xG replaced by Understat's, plus a coverage summary.

    `matches` (the full Understat cache) may be passed in to avoid re-reading it for
    every season; otherwise it is loaded once here.
    """
    matches = matches if matches is not None else ensure_player_matches()
    mapping = build_map(season)
    start_year = _season_start_year(season)

    # Index only the players we actually mapped, once.
    by_player_day = {}
    for element, understat_id in mapping.items():
        by_player_day[element] = _matches_by_date(
            matches.get(understat_id, []), start_year)

    frame = frame.copy()
    xg = frame["expected_goals"].to_numpy(dtype=float).copy()
    xa = frame["expected_assists"].to_numpy(dtype=float).copy()

    covered_minutes = 0
    played_minutes = 0
    covered_rows = 0
    played_rows = 0
    for i, (element, day, minutes) in enumerate(zip(
            frame["player_id"], frame["kickoff_time"], frame["minutes"])):
        if minutes > 0:
            played_minutes += int(minutes)
            played_rows += 1
        match = None
        if element in by_player_day:
            match = _lookup(by_player_day[element], _naive_day(day))
        if match is not None:
            xg[i] = float(match["xG"])
            xa[i] = float(match["xA"])
            if minutes > 0:
                covered_minutes += int(minutes)
                covered_rows += 1
        else:
            xg[i] = 0.0
            xa[i] = 0.0

    frame["expected_goals"] = xg
    frame["expected_assists"] = xa

    summary = {
        "season": season,
        "mapped_players": len(mapping),
        "row_coverage": covered_rows / played_rows if played_rows else float("nan"),
        "minute_coverage": covered_minutes / played_minutes if played_minutes else float("nan"),
    }
    logger.info("%s: Understat xG covers %.1f%% of played minutes (%d mapped players)",
                season, 100 * summary["minute_coverage"], summary["mapped_players"])
    return frame, summary
