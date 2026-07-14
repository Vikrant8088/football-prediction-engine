"""Project a player's expected FPL points for a specific fixture.

This is where the whole project pays off in a market that is actually winnable.
There is no bookmaker in FPL and therefore no margin to overcome - you compete
against other managers' intuition. Our benchmarked team model supplies exactly
the quantities intuition is worst at:

    from the champion's scoreline grid, for THIS fixture:
        E[team goals]          -> scales a player's goal and assist rates
        P(opponent scores 0)   -> clean-sheet points, the hardest thing to eyeball
        P(opponent scores k)   -> the goals-conceded penalty, properly averaged
        E[opponent goals]      -> a goalkeeper's save volume

    from the FPL API, per player:
        xG/90, xA/90 (Opta), saves/90, defensive actions/90, bonus/90, cards/90
        minutes, availability, position, price

Expected points are then the scoring rules (`scoring.py`, validated to 100% on
2,085 real matches) applied to those expectations.

Modelled vs empirical - stated honestly. Appearance, goals, assists, clean
sheets, goals conceded and saves are *modelled* from the fixture. Bonus, cards
and defensive contribution depend on in-match events we do not model, so they
use the player's own realised per-90 rate. They are flagged separately in the
output rather than hidden inside one number.

Approximations, all deliberate and visible:
  - Expected minutes = season minutes / gameweeks played. A player's rotation
    risk is assumed to continue.
  - P(60+ minutes) ~= min(1, expected_minutes / 60). This is exact at both ends
    (an ever-present gets 2 appearance points; a player averaging 30 minutes
    gets ~1) and monotone in between.
  - Defensive-contribution points use P(actions >= threshold) under a Poisson
    with the player's own per-90 rate, rather than a linear rate - because the
    rule is a per-match threshold, not a tally.
  - A fixture multiplier scales a player's scoring rate by how good this fixture
    is for his team, relative to that team's own season average.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import poisson

from prediction_engine.fpl import scoring
from prediction_engine.fpl.minutes import DEFAULT_HALF_LIFE, recent_form_minutes
from prediction_engine.fpl.scoring import (
    APPEARANCE_LONG,
    APPEARANCE_SHORT,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    CONCEDED_PENALTY_POSITIONS,
    CONCEDED_PER_MINUS_ONE,
    DEFENSIVE_CONTRIBUTION_POINTS,
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    GKP,
    GOAL_POINTS,
    MINUTES_FOR_LONG_APPEARANCE,
    SAVES_PER_POINT,
)

logger = logging.getLogger(__name__)

GAMEWEEKS_PER_SEASON = 38
FULL_MATCH_MINUTES = 90
APPEARANCE_MAX_POINTS = 2


def team_scoring_rates(matches: pd.DataFrame, season: str = None) -> Dict[str, Dict[str, float]]:
    """Each team's mean goals scored and conceded per match, in `season`
    (default: the most recent). The denominator for the fixture multiplier."""
    season = season or matches["season"].max()
    recent = matches[matches["season"] == season]

    rates = {}
    for team in set(recent["home_team"]) | set(recent["away_team"]):
        home = recent[recent["home_team"] == team]
        away = recent[recent["away_team"] == team]
        played = len(home) + len(away)
        if played == 0:
            continue
        scored = home["home_goals"].sum() + away["away_goals"].sum()
        conceded = home["away_goals"].sum() + away["home_goals"].sum()
        rates[team] = {
            "scored_per_match": float(scored) / played,
            "conceded_per_match": float(conceded) / played,
        }
    return rates


def fixture_context(grid: np.ndarray, is_home: bool) -> Dict[str, float]:
    """Everything a projection needs about one SIDE of a fixture, read off the
    champion's scoreline grid."""
    goals = np.arange(grid.shape[0])
    if is_home:
        own_goal_dist = grid.sum(axis=1)      # P(this team scores k)
        opp_goal_dist = grid.sum(axis=0)      # P(opponent scores k)
    else:
        own_goal_dist = grid.sum(axis=0)
        opp_goal_dist = grid.sum(axis=1)

    expected_conceded_penalty = float(
        sum(p * (k // CONCEDED_PER_MINUS_ONE) for k, p in enumerate(opp_goal_dist))
    )
    return {
        "expected_goals_for": float((own_goal_dist * goals).sum()),
        "expected_goals_against": float((opp_goal_dist * goals).sum()),
        "clean_sheet_probability": float(opp_goal_dist[0]),
        "expected_conceded_penalty": expected_conceded_penalty,
    }


def expected_minutes(player: pd.Series, gameweeks: int = GAMEWEEKS_PER_SEASON) -> float:
    """Minutes we expect this player to be on the pitch, adjusted for reported
    availability (an injured player's chance_of_playing scales him down)."""
    base = float(player["minutes"]) / max(gameweeks, 1)
    availability = float(player["chance_of_playing"]) / 100.0
    if not bool(player["available"]):
        availability = min(availability, float(player["chance_of_playing"]) / 100.0)
    return float(np.clip(base * availability, 0.0, FULL_MATCH_MINUTES))


def project_player(
    player: pd.Series,
    context: Dict[str, float],
    team_rate: Dict[str, float],
    gameweeks: int = GAMEWEEKS_PER_SEASON,
    minutes_model: Dict[str, float] = None,
) -> Dict[str, float]:
    """Expected FPL points for one player in one fixture, broken into its parts.

    `gameweeks` is the denominator for expected minutes - the number of
    gameweeks the player's `minutes` total was accumulated over. It defaults to a
    full season; a walk-forward backtest passes the gameweeks completed so far.

    `minutes_model` optionally supplies {expected_minutes, p_60, p_play} from a
    richer minutes model (see `minutes.py`). Left None it reproduces the shipped
    flat-average behaviour exactly: p_60 = p_play = min(1, mean/60), so the
    appearance term below collapses to the historical `2 * p_60`.
    """
    position = int(player["position"])
    if minutes_model is None:
        mean = expected_minutes(player, gameweeks=gameweeks)
        p = float(min(1.0, mean / MINUTES_FOR_LONG_APPEARANCE)) if mean > 0 else 0.0
        minutes_model = {"expected_minutes": mean, "p_60": p, "p_play": p}

    minutes = float(minutes_model["expected_minutes"])
    if minutes <= 0:
        return {"expected_points": 0.0, "expected_minutes": 0.0, "p_60_minutes": 0.0,
                "appearance": 0.0, "goals": 0.0, "assists": 0.0, "clean_sheet": 0.0,
                "conceded": 0.0, "saves": 0.0, "bonus": 0.0, "defensive": 0.0, "cards": 0.0}

    minutes_share = minutes / FULL_MATCH_MINUTES
    # p_60 gates clean-sheet points (FPL requires a genuine 60+ appearance) and
    # the long half of the appearance award; p_play covers the 1-point cameo.
    p_60 = float(minutes_model["p_60"])
    p_play = float(minutes_model["p_play"])

    # How good is this fixture for the team, versus its own season average?
    attack_multiplier = (
        context["expected_goals_for"] / team_rate["scored_per_match"]
        if team_rate["scored_per_match"] > 0 else 1.0
    )
    defence_multiplier = (
        context["expected_goals_against"] / team_rate["conceded_per_match"]
        if team_rate["conceded_per_match"] > 0 else 1.0
    )

    exp_goals = float(player["xg_per_90"]) * minutes_share * attack_multiplier
    exp_assists = float(player["xa_per_90"]) * minutes_share * attack_multiplier

    # E[appearance] = 1 * P(1-59 min) + 2 * P(60+ min) = (p_play - p_60) + 2*p_60.
    # With the crude model (p_play == p_60) this is exactly the historical 2*p_60.
    appearance = APPEARANCE_SHORT * max(0.0, p_play - p_60) + APPEARANCE_LONG * p_60
    goals_pts = exp_goals * GOAL_POINTS[position]
    assists_pts = exp_assists * ASSIST_POINTS
    clean_sheet_pts = context["clean_sheet_probability"] * p_60 * CLEAN_SHEET_POINTS[position]

    conceded_pts = 0.0
    if position in CONCEDED_PENALTY_POSITIONS:
        conceded_pts = -context["expected_conceded_penalty"] * minutes_share

    saves_pts = 0.0
    if position == GKP:
        exp_saves = float(player["saves_per_90"]) * minutes_share * defence_multiplier
        saves_pts = exp_saves / SAVES_PER_POINT

    # Threshold rule, so model the count and take P(count >= threshold).
    defensive_pts = 0.0
    threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD.get(position)
    if threshold is not None and player["dc_per_90"] > 0:
        expected_actions = float(player["dc_per_90"]) * minutes_share
        p_threshold = float(poisson.sf(threshold - 1, expected_actions))
        defensive_pts = DEFENSIVE_CONTRIBUTION_POINTS * p_threshold

    bonus_pts = float(player["bonus_per_90"]) * minutes_share
    cards_pts = -float(player["cards_per_90"]) * minutes_share

    total = (
        appearance + goals_pts + assists_pts + clean_sheet_pts
        + conceded_pts + saves_pts + defensive_pts + bonus_pts + cards_pts
    )
    return {
        "expected_points": total,
        "expected_minutes": minutes,
        "appearance_factor": p_60,
        "appearance": appearance,
        "goals": goals_pts,
        "assists": assists_pts,
        "clean_sheet": clean_sheet_pts,
        "conceded": conceded_pts,
        "saves": saves_pts,
        "bonus": bonus_pts,
        "defensive": defensive_pts,
        "cards": cards_pts,
        "expected_goals": exp_goals,
        "expected_assists": exp_assists,
    }


def _live_minutes_model(player: pd.Series, minutes_history: Dict[int, list]):
    """Build the recent-form minutes model for a live player, or None to fall back
    to the crude flat average.

    Uses the player's current-season per-match minutes (recency-weighted) AND the
    live availability flag - `chance_of_playing` scales a doubtful player down and
    zeroes an injured one. The flag is the single biggest minutes signal and is
    exactly what the historical backtest could NOT see, so the live projection is
    strictly better-informed than the backtested +2.95 pts/GW gain implies.
    """
    if minutes_history is None:
        return None
    sequence = minutes_history.get(int(player["id"]))
    if not sequence:
        return None    # no recent history for this player -> crude fallback
    availability = float(player["chance_of_playing"]) / 100.0
    return recent_form_minutes(sequence, half_life_matches=DEFAULT_HALF_LIFE,
                               availability=availability)


def project_fixture(engine, players: pd.DataFrame, home_team: str, away_team: str,
                    minutes_history: Dict[int, list] = None) -> pd.DataFrame:
    """Expected points for every player in one fixture, best first.

    `minutes_history` maps FPL player id -> current-season per-match minutes. When
    supplied, each player is projected with the Phase 6b recent-form minutes model
    (the proven champion); omitted, it falls back to the crude flat average.
    """
    grid = engine.scoreline_grid(home_team, away_team)
    rates = team_scoring_rates(engine.matches)

    for team in (home_team, away_team):
        if team not in rates:
            raise ValueError(f"No recent scoring rate for '{team}'")

    rows = []
    for is_home, team in ((True, home_team), (False, away_team)):
        context = fixture_context(grid, is_home=is_home)
        squad = players[players["team"] == team]
        if squad.empty:
            raise ValueError(f"No FPL players found for team '{team}'")
        for _, player in squad.iterrows():
            minutes_model = _live_minutes_model(player, minutes_history)
            projection = project_player(player, context, rates[team],
                                        minutes_model=minutes_model)
            rows.append({
                "player": player["web_name"],
                "team": team,
                "position": scoring.POSITION_NAMES[int(player["position"])],
                "price": player["price"],
                "opponent": away_team if is_home else home_team,
                "home": is_home,
                "available": bool(player["available"]),
                "clean_sheet_probability": context["clean_sheet_probability"],
                **projection,
            })

    return pd.DataFrame(rows).sort_values("expected_points", ascending=False).reset_index(drop=True)
