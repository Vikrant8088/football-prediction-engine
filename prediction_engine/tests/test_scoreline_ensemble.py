"""Tests for the ScorelineEnsemble.

The invariant that matters, and the reason this class exists at all:

    the scoreline grid's home/draw/away regions must sum to EXACTLY the
    champion ensemble's P(H), P(D), P(A)

If that ever breaks, every derived market silently contradicts the headline
forecast.
"""

import unittest

import numpy as np
import pandas as pd

from prediction_engine.markets import outcome_probabilities
from prediction_engine.scoreline_ensemble import ScorelineEnsemble, outcome_masks


def _matches(n_seasons=4, per_season=30):
    """A small synthetic league where 'Strong' beats 'Weak'."""
    teams = ["Strong", "Mid", "Weak"]
    rows = []
    strength = {"Strong": (2.6, 3), "Mid": (1.3, 1), "Weak": (0.5, 0)}
    day = 0
    for s in range(n_seasons):
        for i in range(per_season):
            home, away = teams[i % 3], teams[(i + 1) % 3]
            hx, hg = strength[home]
            ax, ag = strength[away]
            day += 1
            rows.append({
                "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=day),
                "season": f"{2015 + s}-{str(2016 + s)[-2:]}",
                "home_team": home, "away_team": away,
                "home_goals": hg, "away_goals": ag,
                "home_xg": hx, "away_xg": ax,
                "result": "H" if hg > ag else ("A" if hg < ag else "D"),
            })
    return pd.DataFrame(rows)


class TestScorelineEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ScorelineEnsemble().fit(_matches())

    def test_grid_is_a_valid_distribution(self):
        grid = self.model.scoreline_grid("Strong", "Weak")
        self.assertAlmostEqual(grid.sum(), 1.0, places=9)
        self.assertTrue((grid >= 0).all())

    def test_grid_marginals_equal_the_champion_1x2(self):
        """The central invariant."""
        for home, away in [("Strong", "Weak"), ("Weak", "Strong"), ("Mid", "Strong")]:
            grid = self.model.scoreline_grid(home, away)
            from_grid = np.array(outcome_probabilities(grid))
            from_champion = self.model.predict_proba(
                pd.DataFrame({"home_team": [home], "away_team": [away]})
            )[0]
            self.assertTrue(
                np.allclose(from_grid, from_champion, atol=1e-9),
                msg=f"{home} vs {away}: grid {from_grid} != champion {from_champion}",
            )

    def test_stronger_team_has_more_home_win_mass(self):
        strong = self.model.scoreline_grid("Strong", "Weak")
        weak = self.model.scoreline_grid("Weak", "Strong")
        home_mask = outcome_masks(strong.shape[0])[0]
        self.assertGreater(strong[home_mask].sum(), weak[home_mask].sum())

    def test_falls_back_when_scoreline_models_get_no_weight(self):
        """The blend really does sometimes put 100% on Elo (it did in the most
        recent EPL fold). The grid must still be valid, and still match the
        champion's 1X2."""
        self.model._ensemble.weights_ = {
            "elo": 1.0, "poisson_xg": 0.0, "dixon_coles_xg": 0.0
        }
        try:
            weights = self.model._grid_weights()
            self.assertAlmostEqual(sum(weights.values()), 1.0)
            self.assertAlmostEqual(weights["poisson_xg"], 0.5)

            grid = self.model.scoreline_grid("Strong", "Weak")
            self.assertAlmostEqual(grid.sum(), 1.0, places=9)
            from_grid = np.array(outcome_probabilities(grid))
            from_champion = self.model.predict_proba(
                pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"]})
            )[0]
            self.assertTrue(np.allclose(from_grid, from_champion, atol=1e-9))
        finally:
            self.model._ensemble.fit(_matches())  # restore real weights

    def test_base_views_expose_each_model_opinion(self):
        views = self.model.base_views("Strong", "Weak")
        self.assertEqual(set(views), {"elo", "poisson_xg", "dixon_coles_xg"})
        for view in views.values():
            self.assertAlmostEqual(float(np.sum(view)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
