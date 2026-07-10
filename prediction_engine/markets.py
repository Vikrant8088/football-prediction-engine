"""Every betting/analysis market, derived from the one scoreline grid.

This is the payoff of predicting a *distribution* rather than a scoreline. Given
P(home_goals=h, away_goals=a) for every (h, a), each market below is arithmetic
on that same table - so all of them are mutually consistent by construction, and
all of them inherit the champion's calibration. Nothing here is separately
"modelled"; nothing can contradict anything else.

Grids are indexed [home_goals, away_goals] and sum to 1.
"""

from typing import Dict, List, Tuple

import numpy as np

from prediction_engine.scoreline_ensemble import outcome_masks


def outcome_probabilities(grid: np.ndarray) -> Tuple[float, float, float]:
    """P(home win), P(draw), P(away win)."""
    home, draw, away = outcome_masks(grid.shape[0])
    return float(grid[home].sum()), float(grid[draw].sum()), float(grid[away].sum())


def most_likely_scorelines(grid: np.ndarray, top_n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
    """The `top_n` most probable exact scores, most likely first."""
    flat_order = np.argsort(grid, axis=None)[::-1][:top_n]
    coords = np.unravel_index(flat_order, grid.shape)
    return [
        ((int(h), int(a)), float(grid[h, a]))
        for h, a in zip(coords[0], coords[1])
    ]


def over_under(grid: np.ndarray, line: float = 2.5) -> Dict[str, float]:
    """P(total goals over `line`) and P(under). A .5 line cannot push."""
    goals = np.arange(grid.shape[0])
    totals = goals[:, None] + goals[None, :]
    over = float(grid[totals > line].sum())
    return {"line": line, "over": over, "under": 1.0 - over}


def both_teams_to_score(grid: np.ndarray) -> Dict[str, float]:
    """P(both teams score at least one) - i.e. every cell with h>=1 and a>=1."""
    yes = float(grid[1:, 1:].sum())
    return {"yes": yes, "no": 1.0 - yes}


def double_chance(grid: np.ndarray) -> Dict[str, float]:
    """The three 'two of three outcomes' markets."""
    p_home, p_draw, p_away = outcome_probabilities(grid)
    return {
        "home_or_draw": p_home + p_draw,
        "home_or_away": p_home + p_away,
        "draw_or_away": p_draw + p_away,
    }


def expected_goals(grid: np.ndarray) -> Dict[str, float]:
    """The mean of each side's goal distribution under the grid."""
    goals = np.arange(grid.shape[0])
    home = float((grid.sum(axis=1) * goals).sum())
    away = float((grid.sum(axis=0) * goals).sum())
    return {"home": home, "away": away, "total": home + away}


def clean_sheet(grid: np.ndarray) -> Dict[str, float]:
    """P(each side concedes nothing)."""
    return {"home": float(grid[:, 0].sum()), "away": float(grid[0, :].sum())}


def all_markets(grid: np.ndarray) -> Dict[str, object]:
    """Everything at once, for a serving layer to hand back in one payload."""
    p_home, p_draw, p_away = outcome_probabilities(grid)
    return {
        "outcome": {"home": p_home, "draw": p_draw, "away": p_away},
        "most_likely_scorelines": most_likely_scorelines(grid),
        "over_under_2_5": over_under(grid, 2.5),
        "over_under_1_5": over_under(grid, 1.5),
        "over_under_3_5": over_under(grid, 3.5),
        "both_teams_to_score": both_teams_to_score(grid),
        "double_chance": double_chance(grid),
        "expected_goals": expected_goals(grid),
        "clean_sheet": clean_sheet(grid),
    }
