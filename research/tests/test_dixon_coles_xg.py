"""Unit tests for the xG-aware Dixon-Coles model.

Beyond the usual validity checks, two structural properties are pinned:
 - with time decay switched off (xi=0), Stage 1 must reduce to exactly the
   PoissonXG strength fit (same objective, uniform weights);
 - the fitted rho is finite and the low-score correction is actually applied
   (predictions differ from a pure xG-Poisson on the same strengths).
"""

import unittest

import numpy as np
import pandas as pd

from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.poisson_xg import PoissonXGModel

COLUMNS = [
    "date", "home_team", "away_team",
    "home_goals", "away_goals", "result", "home_xg", "away_xg",
]


def _result(hg, ag):
    return "H" if hg > ag else ("A" if hg < ag else "D")


def _matches(rows, start="2020-01-01"):
    records = []
    for i, (h, a, hx, hg, ax, ag) in enumerate(rows):
        records.append({
            "date": pd.Timestamp(start) + pd.Timedelta(days=i),
            "home_team": h, "away_team": a,
            "home_goals": hg, "away_goals": ag, "result": _result(hg, ag),
            "home_xg": hx, "away_xg": ax,
        })
    return pd.DataFrame.from_records(records, columns=COLUMNS)


class TestDixonColesXGModel(unittest.TestCase):
    def setUp(self):
        self.train = _matches([
            ("Strong", "Weak", 2.8, 3, 0.4, 0),
            ("Weak", "Strong", 0.5, 0, 2.5, 3),
            ("Strong", "Mid", 2.2, 2, 1.0, 1),
            ("Mid", "Strong", 1.1, 1, 2.4, 2),
            ("Mid", "Weak", 1.8, 2, 0.6, 0),
            ("Weak", "Mid", 0.7, 1, 1.6, 2),
        ] * 6)
        self.fixtures = pd.DataFrame(
            {"home_team": ["Strong", "Weak"], "away_team": ["Weak", "Strong"]}
        )

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DixonColesXGModel().predict_proba(self.fixtures)

    def test_probabilities_are_valid(self):
        model = DixonColesXGModel().fit(self.train)
        probs = model.predict_proba(self.fixtures)
        self.assertEqual(probs.shape, (2, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_stronger_team_favored(self):
        model = DixonColesXGModel().fit(self.train)
        p_home, _, p_away = model.predict_proba(self.fixtures)[0]  # Strong at home
        self.assertGreater(p_home, p_away)

    def test_rho_is_finite_and_small(self):
        model = DixonColesXGModel().fit(self.train)
        self.assertTrue(np.isfinite(model._rho))
        # rho is a small correlation-style correction, not an unbounded blow-up.
        self.assertLess(abs(model._rho), 2.0)

    def test_unseen_team_falls_back_without_error(self):
        model = DixonColesXGModel().fit(self.train)
        fixtures = pd.DataFrame({"home_team": ["Promoted"], "away_team": ["Strong"]})
        probs = model.predict_proba(fixtures)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))

    def test_no_time_decay_reduces_to_poisson_xg_strengths(self):
        """xi=0 => uniform weights => Stage 1 is exactly the PoissonXG fit."""
        dc_xg = DixonColesXGModel(xi=0.0).fit(self.train)
        pois_xg = PoissonXGModel().fit(self.train)
        self.assertTrue(np.allclose(dc_xg._attack, pois_xg._attack, atol=1e-4))
        self.assertTrue(np.allclose(dc_xg._defense, pois_xg._defense, atol=1e-4))
        self.assertAlmostEqual(
            dc_xg._home_advantage, pois_xg._home_advantage, places=4
        )

    def test_low_score_correction_is_applied_in_prediction(self):
        """The tau(rho) correction must be wired into the prediction path: with
        strengths fixed, a non-zero rho moves the scoreline grid versus rho=0.
        (Tested by setting rho directly, since whether the *fit* lands on a
        non-zero rho depends on there being low-score matches in the data.)"""
        model = DixonColesXGModel().fit(self.train)
        model._rho = 0.0
        uncorrected = model.predict_proba(self.fixtures)
        model._rho = 0.3
        corrected = model.predict_proba(self.fixtures)
        self.assertFalse(np.allclose(corrected, uncorrected))


if __name__ == "__main__":
    unittest.main()
