"""Unit tests for FeatureLogisticModel."""

import unittest

import numpy as np
import pandas as pd

from research.experiments.feature_logistic import FeatureLogisticModel


def _data(n=200, seed_pattern=True):
    """Synthetic matches where a single feature `x` cleanly separates the
    outcome: large x -> home win, small x -> away win, middling -> draw."""
    xs = np.linspace(-3, 3, n)
    rows = []
    for x in xs:
        if x > 1:
            result = "H"
        elif x < -1:
            result = "A"
        else:
            result = "D"
        rows.append({"home_team": "H", "away_team": "A", "result": result, "x": x})
    return pd.DataFrame(rows)


class TestFeatureLogisticModel(unittest.TestCase):
    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            FeatureLogisticModel(["x"]).predict_proba(pd.DataFrame({"x": [0.0]}))

    def test_probabilities_are_valid(self):
        model = FeatureLogisticModel(["x"]).fit(_data())
        probs = model.predict_proba(pd.DataFrame({"x": [-2.0, 0.0, 2.0]}))
        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_learns_a_separating_feature(self):
        model = FeatureLogisticModel(["x"]).fit(_data())
        probs = model.predict_proba(pd.DataFrame({"x": [-2.5, 2.5]}))
        # Large x -> home win most likely; small x -> away win most likely.
        self.assertEqual(np.argmax(probs[1]), 0)  # x=2.5 -> H (col 0)
        self.assertEqual(np.argmax(probs[0]), 2)  # x=-2.5 -> A (col 2)

    def test_handles_missing_features_via_training_mean(self):
        model = FeatureLogisticModel(["x"]).fit(_data())
        probs = model.predict_proba(pd.DataFrame({"x": [np.nan]}))  # cold start
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertFalse(np.isnan(probs).any())

    def test_more_features_are_accepted(self):
        df = _data()
        df["y"] = df["x"] * 0.5 + 0.1
        model = FeatureLogisticModel(["x", "y"]).fit(df)
        probs = model.predict_proba(df[["x", "y"]].head())
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()


class TestDoesNotMutateCallerFrame(unittest.TestCase):
    """The design matrix fills missing values in place, so it must own its array.

    Caught by CI on its first run: under pandas copy-on-write `to_numpy()` returns a
    READ-ONLY view, and the fill raised "assignment destination is read-only". On
    older versions the same code silently wrote through into the caller's DataFrame
    instead — a far worse failure, because the training frame would be quietly edited.
    """

    def test_source_frame_is_untouched_by_the_nan_fill(self):
        import numpy as np
        import pandas as pd
        from research.experiments.feature_logistic import FeatureLogisticModel

        df = pd.DataFrame({
            "f1": [1.0, 2.0, np.nan, 4.0],
            "f2": [0.5, np.nan, 1.5, 2.0],
            "result": ["H", "D", "A", "H"],
        })
        before = df.copy()
        FeatureLogisticModel(feature_cols=["f1", "f2"]).fit(df)
        pd.testing.assert_frame_equal(df, before)
        self.assertTrue(df["f1"].isna().any(), "the caller's NaNs must survive")
