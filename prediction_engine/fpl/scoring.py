"""The official Fantasy Premier League scoring rules, encoded exactly.

Everything a fantasy projection produces is a weighted sum of these rules, so
getting them wrong quietly corrupts every number downstream. They are therefore
written once, here, as named constants - and `match_points()` is validated
against thousands of REAL scored matches from the FPL API
(`research/evaluation/validate_fpl_scoring.py`). Nothing is taken on trust.

Rules (2025/26 season):

    appearance          1 pt (played 1-59 min), 2 pts (60+ min)
    goal                GKP/DEF 6, MID 5, FWD 4
    assist              3 (all positions)
    clean sheet         GKP/DEF 4, MID 1, FWD 0   (only if 60+ min)
    saves               1 pt per 3 saves (GKP)
    penalty save        5     penalty miss  -2
    goals conceded      -1 per 2 conceded (GKP/DEF only)
    yellow card         -1    red card      -3    own goal  -2
    bonus               0-3 (from FPL's BPS system)
    defensive contribution   2 pts, at DEF 10+ / MID+FWD 12+ actions   [new in 2025/26]

The defensive-contribution rule is why `defensive_contribution` is a per-MATCH
count: the threshold applies within a single match, so it cannot be recovered
from a season total.
"""

from typing import Dict

# Position codes, as FPL's `element_type` numbers them.
GKP, DEF, MID, FWD = 1, 2, 3, 4
POSITION_NAMES: Dict[int, str] = {GKP: "GKP", DEF: "DEF", MID: "MID", FWD: "FWD"}

APPEARANCE_SHORT = 1          # played 1-59 minutes
APPEARANCE_LONG = 2           # played 60+ minutes
MINUTES_FOR_LONG_APPEARANCE = 60

GOAL_POINTS: Dict[int, int] = {GKP: 6, DEF: 6, MID: 5, FWD: 4}
ASSIST_POINTS = 3
CLEAN_SHEET_POINTS: Dict[int, int] = {GKP: 4, DEF: 4, MID: 1, FWD: 0}

SAVES_PER_POINT = 3           # goalkeepers only
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2

CONCEDED_PER_MINUS_ONE = 2    # goalkeepers and defenders only
CONCEDED_PENALTY_POSITIONS = (GKP, DEF)

YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2

# Defensive contribution (2025/26). A single match must reach the threshold.
DEFENSIVE_CONTRIBUTION_POINTS = 2
DEFENSIVE_CONTRIBUTION_THRESHOLD: Dict[int, int] = {DEF: 10, MID: 12, FWD: 12}


def appearance_points(minutes: int) -> int:
    if minutes <= 0:
        return 0
    return APPEARANCE_LONG if minutes >= MINUTES_FOR_LONG_APPEARANCE else APPEARANCE_SHORT


def defensive_contribution_points(position: int, defensive_contribution: int) -> int:
    """2 pts if the match's defensive-action count clears the position threshold.
    Goalkeepers are not eligible."""
    threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD.get(position)
    if threshold is None:
        return 0
    return DEFENSIVE_CONTRIBUTION_POINTS if defensive_contribution >= threshold else 0


def match_points(
    position: int,
    minutes: int = 0,
    goals_scored: int = 0,
    assists: int = 0,
    clean_sheets: int = 0,
    goals_conceded: int = 0,
    saves: int = 0,
    bonus: int = 0,
    yellow_cards: int = 0,
    red_cards: int = 0,
    own_goals: int = 0,
    penalties_missed: int = 0,
    penalties_saved: int = 0,
    defensive_contribution: int = 0,
) -> int:
    """Total FPL points a player scored in ONE match.

    Argument names deliberately mirror the FPL API's per-match history fields,
    so a history row can be passed straight through as **kwargs.
    """
    if position not in POSITION_NAMES:
        raise ValueError(f"Unknown position {position}; expected one of {sorted(POSITION_NAMES)}")
    if minutes <= 0:
        return 0  # an unused substitute scores nothing

    points = appearance_points(minutes)
    points += goals_scored * GOAL_POINTS[position]
    points += assists * ASSIST_POINTS
    # `clean_sheets` from the API is already 0/1 and already respects the
    # 60-minute requirement, so it is used as given rather than re-derived.
    points += clean_sheets * CLEAN_SHEET_POINTS[position]

    if position == GKP:
        points += saves // SAVES_PER_POINT
        points += penalties_saved * PENALTY_SAVE_POINTS

    if position in CONCEDED_PENALTY_POSITIONS:
        points -= goals_conceded // CONCEDED_PER_MINUS_ONE

    points += defensive_contribution_points(position, defensive_contribution)
    points += bonus
    points += yellow_cards * YELLOW_CARD_POINTS
    points += red_cards * RED_CARD_POINTS
    points += own_goals * OWN_GOAL_POINTS
    points += penalties_missed * PENALTY_MISS_POINTS
    return int(points)


def match_points_from_history(position: int, history_row: dict) -> int:
    """Score one row of the FPL API's per-match `history` list."""
    fields = (
        "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
        "saves", "bonus", "yellow_cards", "red_cards", "own_goals",
        "penalties_missed", "penalties_saved", "defensive_contribution",
    )
    return match_points(position, **{f: int(history_row.get(f, 0) or 0) for f in fields})
