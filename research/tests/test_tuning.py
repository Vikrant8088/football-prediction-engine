"""Unit tests for leakage-safe nested hyperparameter tuning."""

import unittest

import numpy as np
import pandas as pd

from research.experiments.base import PredictionModel
from research.experiments.tuning import TunedModel


class _ConstantModel(PredictionModel):
    """Ignores training; predicts (p_home, rest split evenly). The parameter
    under search is `p_home`, so the tuner's choice is fully predictable."""

    def __init__(self, p_home=0.5):
        self.p_home = p_home

    def fit(self, train_df):
        return self

    def predict_proba(self, fixtures_df):
        rest = (1.0 - self.p_home) / 2.0
        return np.tile([self.p_home, rest, rest], (len(fixtures_df), 1))


def _season(label, result, n=10):
    return [
        {
            "season": label,
            "home_team": f"T{i % 3}",
            "away_team": f"T{(i + 1) % 3}",
            "result": result,
        }
        for i in range(n)
    ]


GRID = {"p_home": [0.1, 0.9]}
DEFAULTS = {"p_home": 0.5}


class TestTunedModel(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            TunedModel(_ConstantModel, GRID, DEFAULTS).predict_proba(
                pd.DataFrame({"home_team": ["a"], "away_team": ["b"]})
            )

    def test_selects_params_on_the_inner_validation_seasons(self):
        """The two most recent training seasons are all home wins; the earlier
        two are all away wins. A tuner that validates on the RECENT seasons must
        pick p_home=0.9. Picking 0.1 would mean it scored on the older seasons
        (or on everything), which is the behaviour this guards against."""
        df = pd.DataFrame(
            _season("2018-19", "A") + _season("2019-20", "A")
            + _season("2020-21", "H") + _season("2021-22", "H")
        )
        model = TunedModel(_ConstantModel, GRID, DEFAULTS, inner_val_seasons=2).fit(df)
        self.assertEqual(model.best_params_["p_home"], 0.9)

    def test_falls_back_to_defaults_without_enough_seasons(self):
        df = pd.DataFrame(_season("2020-21", "H") + _season("2021-22", "H"))
        model = TunedModel(_ConstantModel, GRID, DEFAULTS, inner_val_seasons=2).fit(df)
        self.assertEqual(model.best_params_, DEFAULTS)

    def test_predictions_are_valid_and_use_the_chosen_params(self):
        df = pd.DataFrame(
            _season("2018-19", "A") + _season("2019-20", "A")
            + _season("2020-21", "H") + _season("2021-22", "H")
        )
        model = TunedModel(_ConstantModel, GRID, DEFAULTS, inner_val_seasons=2).fit(df)
        probs = model.predict_proba(pd.DataFrame({"home_team": ["a"], "away_team": ["b"]}))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertAlmostEqual(probs[0][0], 0.9)  # the tuned p_home

    def test_grid_is_searched_exhaustively(self):
        seen = []

        class _Spy(_ConstantModel):
            def __init__(self, p_home=0.5, other=1):
                super().__init__(p_home)
                seen.append((p_home, other))

        df = pd.DataFrame(
            _season("2018-19", "H") + _season("2019-20", "H")
            + _season("2020-21", "H") + _season("2021-22", "H")
        )
        grid = {"p_home": [0.1, 0.9], "other": [1, 2]}
        TunedModel(_Spy, grid, {"p_home": 0.5, "other": 1}, inner_val_seasons=2).fit(df)
        # 2 x 2 candidates evaluated (plus one final refit with the winner).
        self.assertEqual(len(set(seen)), 4)


if __name__ == "__main__":
    unittest.main()
