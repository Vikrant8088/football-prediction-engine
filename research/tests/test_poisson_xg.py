"""Unit tests for the Phase 1 xG scoreline model (PoissonXGModel).

The headline behaviour under test is the one thing that distinguishes this
model from the goal-based Poisson: it must estimate team strength from the
`home_xg`/`away_xg` columns, NOT from goals. The strongest test constructs a
dataset where xG and goals disagree by construction and asserts the two models
rank a team in OPPOSITE directions - proof the xG model really uses xG.
"""

import unittest

import numpy as np
import pandas as pd

from research.experiments.poisson import PoissonModel
from research.experiments.poisson_xg import PoissonXGModel

COLUMNS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals", "result", "home_xg", "away_xg",
]


def _result(hg, ag):
    return "H" if hg > ag else ("A" if hg < ag else "D")


def _matches(rows):
    """rows: list of (home, away, home_xg, home_goals, away_xg, away_goals)."""
    records = []
    for i, (h, a, hx, hg, ax, ag) in enumerate(rows):
        records.append({
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag, "result": _result(hg, ag),
            "home_xg": hx, "away_xg": ax,
        })
    return pd.DataFrame.from_records(records, columns=COLUMNS)


# Per-team attacking profile where xG and actual goals DISAGREE on purpose:
#   Unlucky - creates lots of chances (high xG) but scores few goals
#   Lucky   - creates few chances (low xG) but scores many goals
#   Aref/Mid- neutral (Aref sorts first, so it is the pinned reference team)
_PROFILE = {
    "Unlucky": {"xg": 2.6, "goals": 0},
    "Lucky": {"xg": 0.4, "goals": 3},
    "Aref": {"xg": 1.2, "goals": 1},
    "Mid": {"xg": 1.2, "goals": 1},
}


def _divergent_dataset(repeats=8):
    teams = list(_PROFILE)
    rows = []
    for _ in range(repeats):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                rows.append((
                    h, a,
                    _PROFILE[h]["xg"], _PROFILE[h]["goals"],
                    _PROFILE[a]["xg"], _PROFILE[a]["goals"],
                ))
    return _matches(rows)


class TestPoissonXGModel(unittest.TestCase):
    def setUp(self):
        # A simple, well-separated league for the validity tests.
        self.train = _matches([
            ("Strong", "Weak", 2.8, 3, 0.4, 0),
            ("Weak", "Strong", 0.5, 0, 2.5, 3),
            ("Strong", "Mid", 2.2, 2, 1.0, 1),
            ("Mid", "Strong", 1.1, 1, 2.4, 2),
            ("Mid", "Weak", 1.8, 2, 0.6, 0),
            ("Weak", "Mid", 0.7, 1, 1.6, 2),
        ] * 6)

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            PoissonXGModel().predict_proba(pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"]}))

    def test_probabilities_are_valid(self):
        model = PoissonXGModel().fit(self.train)
        fixtures = pd.DataFrame({
            "home_team": ["Strong", "Weak", "Mid"],
            "away_team": ["Weak", "Strong", "Weak"],
        })
        probs = model.predict_proba(fixtures)
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_reference_team_is_pinned(self):
        model = PoissonXGModel().fit(self.train)
        # Identifiability: the first (sorted) team's attack & defense are 0.
        self.assertAlmostEqual(model._attack[0], 0.0, places=9)
        self.assertAlmostEqual(model._defense[0], 0.0, places=9)

    def test_stronger_team_more_likely_to_win(self):
        model = PoissonXGModel().fit(self.train)
        fixtures = pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"]})
        p_home, p_draw, p_away = model.predict_proba(fixtures)[0]
        self.assertGreater(p_home, p_away)
        self.assertGreater(p_home, 0.5)

    def test_unseen_team_falls_back_without_error(self):
        model = PoissonXGModel().fit(self.train)
        fixtures = pd.DataFrame({"home_team": ["Promoted"], "away_team": ["Strong"]})
        probs = model.predict_proba(fixtures)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_model_uses_xg_not_goals(self):
        """The defining test: on data where xG and goals disagree, the xG model
        and the goal model must rank Unlucky vs Lucky in OPPOSITE directions."""
        data = _divergent_dataset()
        xg_model = PoissonXGModel().fit(data)
        goal_model = PoissonModel().fit(data)

        xg_idx = xg_model._team_idx
        g_idx = goal_model._team_idx

        # xG model: Unlucky (high xG) rated a stronger attacker than Lucky.
        self.assertGreater(
            xg_model._attack[xg_idx["Unlucky"]],
            xg_model._attack[xg_idx["Lucky"]],
        )
        # Goal model: the ranking flips - Lucky (high goals) looks stronger.
        self.assertGreater(
            goal_model._attack[g_idx["Lucky"]],
            goal_model._attack[g_idx["Unlucky"]],
        )

    def test_fit_is_deterministic(self):
        a = PoissonXGModel().fit(self.train).predict_proba(
            pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"]})
        )
        b = PoissonXGModel().fit(self.train).predict_proba(
            pd.DataFrame({"home_team": ["Strong"], "away_team": ["Weak"]})
        )
        self.assertTrue(np.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
