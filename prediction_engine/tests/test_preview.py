"""The match-preview numbers must read the grid correctly — a wrong % in a public
post is the one thing the brand can't afford. Pinned on synthetic grids so the maths
is exact and independent of any fitted model."""

import unittest

import numpy as np

from prediction_engine.preview import (match_preview, render_post, resolve_league)


class _FixedEngine:
    def __init__(self, grid):
        self._grid = grid

    def scoreline_grid(self, home, away, allow_unseen=False):
        return self._grid


def _grid(cells, size=6):
    g = np.zeros((size, size))
    for (h, a), p in cells.items():
        g[h, a] = p
    return g / g.sum()


class TestLeagueResolve(unittest.TestCase):
    def test_friendly_names_map_to_understat_codes(self):
        self.assertEqual(resolve_league("laliga"), "La_liga")
        self.assertEqual(resolve_league("La Liga"), "La_liga")
        self.assertEqual(resolve_league("seriea"), "Serie_A")
        self.assertEqual(resolve_league("EPL"), "EPL")
        self.assertEqual(resolve_league("bundesliga"), "Bundesliga")

    def test_unknown_league_raises_with_options(self):
        with self.assertRaises(ValueError):
            resolve_league("mls")


class TestPreviewMaths(unittest.TestCase):
    def test_outcome_probabilities_read_the_grid(self):
        # Home 2-0 half the time, 1-1 the other half: home win 50%, draw 50%.
        pv = match_preview(_FixedEngine(_grid({(2, 0): 0.5, (1, 1): 0.5})), "H", "A")
        self.assertAlmostEqual(pv["p_home"], 0.5)
        self.assertAlmostEqual(pv["p_draw"], 0.5)
        self.assertAlmostEqual(pv["p_away"], 0.0)

    def test_expected_goals_and_likeliest_score(self):
        pv = match_preview(_FixedEngine(_grid({(2, 0): 0.5, (1, 1): 0.5})), "H", "A")
        self.assertAlmostEqual(pv["exp_home"], 1.5)   # (2+1)/2
        self.assertAlmostEqual(pv["exp_away"], 0.5)   # (0+1)/2
        self.assertIn(pv["likeliest"], [(2, 0), (1, 1)])

    def test_btts_and_over_25(self):
        # 2-1 always: both score, and 3 goals is over 2.5.
        pv = match_preview(_FixedEngine(_grid({(2, 1): 1.0})), "H", "A")
        self.assertAlmostEqual(pv["btts"], 1.0)
        self.assertAlmostEqual(pv["over_25"], 1.0)
        # 1-0 always: no BTTS, under 2.5.
        pv = match_preview(_FixedEngine(_grid({(1, 0): 1.0})), "H", "A")
        self.assertAlmostEqual(pv["btts"], 0.0)
        self.assertAlmostEqual(pv["over_25"], 0.0)

    def test_probabilities_sum_to_one(self):
        pv = match_preview(_FixedEngine(_grid({(0, 0): 0.2, (1, 0): 0.3, (0, 2): 0.5})),
                           "H", "A")
        self.assertAlmostEqual(pv["p_home"] + pv["p_draw"] + pv["p_away"], 1.0)


class TestRenderPost(unittest.TestCase):
    def test_post_carries_the_teams_and_the_before_kickoff_line(self):
        pv = match_preview(_FixedEngine(_grid({(2, 1): 0.6, (1, 1): 0.4})),
                           "Barcelona", "Real Madrid")
        post = render_post(pv, "La_liga", handle="@thelockerroomco")
        self.assertIn("Barcelona", post)
        self.assertIn("Real Madrid", post)
        self.assertIn("before kickoff", post)
        self.assertIn("@thelockerroomco", post)
        self.assertIn("#LaLiga", post)


if __name__ == "__main__":
    unittest.main()
