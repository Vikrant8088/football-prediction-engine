"""Unit tests for the weighted EnsembleModel.

The behaviours that matter: it produces valid probabilities, its weights form
a proper distribution, it up-weights the base model that predicts better, a
blend of identical models is that model, and it slots into the walk-forward
harness like any other model.
"""

import unittest

import numpy as np
import pandas as pd

from research.experiments.base import PredictionModel
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.ensemble import EnsembleModel
from research.evaluation.benchmark import run_walk_forward


class _ConstantModel(PredictionModel):
    """A stub base model that ignores training and always predicts a fixed
    (H, D, A) distribution - lets tests control base behaviour exactly."""

    def __init__(self, probs):
        self._p = np.asarray(probs, dtype=float)

    def fit(self, train_df):
        return self

    def predict_proba(self, fixtures_df):
        return np.tile(self._p, (len(fixtures_df), 1))


def _matches(n_seasons, result="H", per_season=10):
    rows = []
    for s in range(n_seasons):
        for i in range(per_season):
            rows.append({
                "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=365 * s + i),
                "season": f"{2015 + s}-{str(2016 + s)[-2:]}",
                "home_team": f"T{i % 4}",
                "away_team": f"T{(i + 1) % 4}",
                "result": result,
            })
    return pd.DataFrame(rows)


class TestEnsembleModel(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        ens = EnsembleModel({"a": lambda: _ConstantModel([0.4, 0.3, 0.3])})
        with self.assertRaises(RuntimeError):
            ens.predict_proba(pd.DataFrame({"home_team": ["x"], "away_team": ["y"]}))

    def test_weights_form_a_distribution(self):
        ens = EnsembleModel({
            "a": lambda: _ConstantModel([0.6, 0.2, 0.2]),
            "b": lambda: _ConstantModel([0.2, 0.4, 0.4]),
        }, val_seasons=2)
        ens.fit(_matches(4))
        self.assertEqual(set(ens.weights_), {"a", "b"})
        self.assertAlmostEqual(sum(ens.weights_.values()), 1.0, places=9)
        self.assertTrue(all(w >= 0 for w in ens.weights_.values()))

    def test_probabilities_are_valid(self):
        ens = EnsembleModel({
            "a": lambda: _ConstantModel([0.6, 0.2, 0.2]),
            "b": lambda: _ConstantModel([0.2, 0.4, 0.4]),
        }).fit(_matches(4))
        probs = ens.predict_proba(pd.DataFrame({
            "home_team": ["A", "B"], "away_team": ["B", "A"],
        }))
        self.assertEqual(probs.shape, (2, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_upweights_the_better_base_model(self):
        # Every match is a home win. The base that says so confidently must get
        # more weight than the one that bets against it.
        ens = EnsembleModel({
            "good": lambda: _ConstantModel([0.8, 0.1, 0.1]),
            "bad": lambda: _ConstantModel([0.1, 0.1, 0.8]),
        }, val_seasons=2).fit(_matches(4, result="H"))
        self.assertGreater(ens.weights_["good"], ens.weights_["bad"])

    def test_blend_of_identical_models_is_that_model(self):
        p = [0.5, 0.3, 0.2]
        ens = EnsembleModel({
            "a": lambda: _ConstantModel(p),
            "b": lambda: _ConstantModel(p),
        }).fit(_matches(4))
        out = ens.predict_proba(pd.DataFrame({"home_team": ["x"], "away_team": ["y"]}))
        self.assertTrue(np.allclose(out[0], p))

    def test_falls_back_to_equal_weights_without_enough_seasons(self):
        ens = EnsembleModel({
            "a": lambda: _ConstantModel([0.6, 0.2, 0.2]),
            "b": lambda: _ConstantModel([0.2, 0.4, 0.4]),
        }, val_seasons=2).fit(_matches(1))  # 1 season <= val_seasons
        self.assertAlmostEqual(ens.weights_["a"], 0.5, places=9)
        self.assertAlmostEqual(ens.weights_["b"], 0.5, places=9)

    def test_integrates_with_walk_forward_harness(self):
        matches = _matches(4, result="H", per_season=12)
        builders = {
            "baseline": BaselineFrequencyModel,
            "ensemble": lambda: EnsembleModel({
                "baseline": BaselineFrequencyModel,
                "const": lambda: _ConstantModel([0.5, 0.25, 0.25]),
            }),
        }
        predictions, runtimes = run_walk_forward(
            matches, model_builders=builders, min_training_seasons=3
        )
        self.assertIn("ensemble", set(predictions["model"]))
        probs = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
