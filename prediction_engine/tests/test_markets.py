"""Tests for markets derived from a scoreline grid.

Grids are hand-built so every expected number can be computed by hand.
"""

import unittest

import numpy as np

from prediction_engine.markets import (
    all_markets,
    both_teams_to_score,
    clean_sheet,
    double_chance,
    expected_goals,
    most_likely_scorelines,
    outcome_probabilities,
    over_under,
)


def _grid(cells: dict, size: int = 4) -> np.ndarray:
    g = np.zeros((size, size))
    for (h, a), p in cells.items():
        g[h, a] = p
    assert abs(g.sum() - 1.0) < 1e-9, "test grid must sum to 1"
    return g


class TestMarkets(unittest.TestCase):
    def setUp(self):
        # 1-0 (home win), 1-1 (draw), 0-2 (away win), 2-1 (home win), 0-0 (draw)
        self.grid = _grid({
            (1, 0): 0.30,
            (1, 1): 0.20,
            (0, 2): 0.15,
            (2, 1): 0.25,
            (0, 0): 0.10,
        })

    def test_outcome_probabilities(self):
        home, draw, away = outcome_probabilities(self.grid)
        self.assertAlmostEqual(home, 0.55)   # 1-0 + 2-1
        self.assertAlmostEqual(draw, 0.30)   # 1-1 + 0-0
        self.assertAlmostEqual(away, 0.15)   # 0-2
        self.assertAlmostEqual(home + draw + away, 1.0)

    def test_most_likely_scorelines_ordered(self):
        top = most_likely_scorelines(self.grid, top_n=3)
        self.assertEqual(top[0][0], (1, 0))
        self.assertAlmostEqual(top[0][1], 0.30)
        self.assertEqual(top[1][0], (2, 1))
        self.assertEqual(top[2][0], (1, 1))
        # strictly non-increasing
        probs = [p for _, p in top]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_over_under_2_5(self):
        # totals: 1-0=1, 1-1=2, 0-2=2, 2-1=3, 0-0=0 -> only 2-1 is over 2.5
        ou = over_under(self.grid, 2.5)
        self.assertAlmostEqual(ou["over"], 0.25)
        self.assertAlmostEqual(ou["under"], 0.75)
        self.assertAlmostEqual(ou["over"] + ou["under"], 1.0)

    def test_over_under_1_5(self):
        # over 1.5 -> totals >= 2: 1-1, 0-2, 2-1 = 0.20 + 0.15 + 0.25
        self.assertAlmostEqual(over_under(self.grid, 1.5)["over"], 0.60)

    def test_both_teams_to_score(self):
        # both score: 1-1 and 2-1
        btts = both_teams_to_score(self.grid)
        self.assertAlmostEqual(btts["yes"], 0.45)
        self.assertAlmostEqual(btts["no"], 0.55)

    def test_double_chance_matches_outcomes(self):
        dc = double_chance(self.grid)
        self.assertAlmostEqual(dc["home_or_draw"], 0.85)
        self.assertAlmostEqual(dc["home_or_away"], 0.70)
        self.assertAlmostEqual(dc["draw_or_away"], 0.45)

    def test_expected_goals(self):
        # home: 1*.30 + 1*.20 + 0*.15 + 2*.25 + 0*.10 = 1.0
        # away: 0*.30 + 1*.20 + 2*.15 + 1*.25 + 0*.10 = 0.75
        eg = expected_goals(self.grid)
        self.assertAlmostEqual(eg["home"], 1.00)
        self.assertAlmostEqual(eg["away"], 0.75)
        self.assertAlmostEqual(eg["total"], 1.75)

    def test_clean_sheet(self):
        cs = clean_sheet(self.grid)
        self.assertAlmostEqual(cs["home"], 0.40)  # away scored 0: 1-0, 0-0
        self.assertAlmostEqual(cs["away"], 0.25)  # home scored 0: 0-2, 0-0

    def test_all_markets_is_internally_consistent(self):
        m = all_markets(self.grid)
        o = m["outcome"]
        self.assertAlmostEqual(o["home"] + o["draw"] + o["away"], 1.0)
        self.assertAlmostEqual(
            m["double_chance"]["home_or_draw"], o["home"] + o["draw"]
        )
        self.assertAlmostEqual(
            m["over_under_2_5"]["over"] + m["over_under_2_5"]["under"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
