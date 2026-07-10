"""Tests for selective prediction / confidence tiers."""

import unittest

import numpy as np
import pandas as pd

from prediction_engine.confidence import (
    DEFAULT_THRESHOLD,
    TIERS,
    classify,
    coverage_at,
    is_confident,
    recompute_tiers,
)


class TestConfidence(unittest.TestCase):
    def test_tiers_are_contiguous_and_cover_zero_to_one(self):
        self.assertAlmostEqual(TIERS[0][0], 0.0)
        for (_, high, _, _, _), (low, _, _, _, _) in zip(TIERS, TIERS[1:]):
            self.assertAlmostEqual(high, low)  # no gaps, no overlaps
        self.assertGreaterEqual(TIERS[-1][1], 1.0)

    def test_measured_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(t[4] for t in TIERS), 1.0, places=2)

    def test_accuracy_increases_with_confidence(self):
        accuracies = [t[3] for t in TIERS]
        self.assertEqual(accuracies, sorted(accuracies))

    def test_classify_picks_the_right_tier(self):
        self.assertEqual(classify(0.35)["tier"], "very low")
        self.assertEqual(classify(0.55)["tier"], "medium")
        self.assertEqual(classify(0.65)["tier"], "high")
        self.assertEqual(classify(0.90)["tier"], "very high")

    def test_classify_attaches_backtested_accuracy(self):
        result = classify(0.65)
        self.assertAlmostEqual(result["backtested_accuracy"], 0.651)
        self.assertIn("measurement_run", result)

    def test_classify_rejects_impossible_probability(self):
        with self.assertRaises(ValueError):
            classify(1.5)

    def test_is_confident_respects_threshold(self):
        self.assertTrue(is_confident(0.61, 0.60))
        self.assertTrue(is_confident(0.60, 0.60))  # boundary is inclusive
        self.assertFalse(is_confident(0.59, 0.60))

    def test_coverage_falls_as_the_threshold_rises(self):
        self.assertGreater(coverage_at(0.5), coverage_at(0.6))
        self.assertGreater(coverage_at(0.6), coverage_at(0.7))
        # At the default threshold we call roughly a quarter of matches.
        self.assertAlmostEqual(coverage_at(DEFAULT_THRESHOLD), 0.254, places=2)

    def test_recompute_tiers_from_predictions(self):
        # Two very confident, correct home picks; two low-confidence wrong ones.
        predictions = pd.DataFrame([
            {"model": "ensemble", "result": "H", "p_home": 0.80, "p_draw": 0.1, "p_away": 0.10},
            {"model": "ensemble", "result": "H", "p_home": 0.75, "p_draw": 0.15, "p_away": 0.10},
            {"model": "ensemble", "result": "A", "p_home": 0.38, "p_draw": 0.32, "p_away": 0.30},
            {"model": "ensemble", "result": "D", "p_home": 0.39, "p_draw": 0.31, "p_away": 0.30},
        ])
        table = recompute_tiers(predictions)
        very_high = table[table["tier"] == "very high"].iloc[0]
        self.assertEqual(very_high["n_matches"], 2)
        self.assertAlmostEqual(very_high["accuracy"], 1.0)
        very_low = table[table["tier"] == "very low"].iloc[0]
        self.assertAlmostEqual(very_low["accuracy"], 0.0)

    def test_recompute_ignores_other_models(self):
        predictions = pd.DataFrame([
            {"model": "ensemble", "result": "H", "p_home": 0.80, "p_draw": 0.1, "p_away": 0.10},
            {"model": "elo", "result": "A", "p_home": 0.90, "p_draw": 0.05, "p_away": 0.05},
        ])
        table = recompute_tiers(predictions, model="ensemble")
        self.assertEqual(int(table["n_matches"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
