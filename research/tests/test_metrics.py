"""Unit tests for the proper scoring rules that underpin every benchmark
result. If these are wrong, the headline "elo wins / xG beats goals" numbers
are meaningless - so they get known-value and property-based checks.
"""

import math
import unittest

import numpy as np

from research.evaluation.metrics import (
    brier_score,
    evaluate_all,
    log_loss,
    ranked_probability_score,
)


class TestMetrics(unittest.TestCase):
    def test_log_loss_uniform_is_log_three(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        self.assertAlmostEqual(log_loss(probs, ["H"]), math.log(3), places=6)

    def test_log_loss_confident_correct_near_zero(self):
        probs = np.array([[0.99, 0.005, 0.005]])
        self.assertLess(log_loss(probs, ["H"]), 0.02)

    def test_log_loss_punishes_confident_wrong(self):
        correct = log_loss(np.array([[0.99, 0.005, 0.005]]), ["H"])
        wrong = log_loss(np.array([[0.005, 0.005, 0.99]]), ["H"])
        self.assertGreater(wrong, correct * 100)

    def test_log_loss_handles_zero_probability_without_inf(self):
        # A model can assign 0 to the outcome that happens; the metric must
        # clip rather than return inf/nan.
        value = log_loss(np.array([[0.0, 0.0, 1.0]]), ["H"])
        self.assertTrue(math.isfinite(value))

    def test_brier_known_value(self):
        probs = np.array([[1 / 3, 1 / 3, 1 / 3]])
        # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 6/9
        self.assertAlmostEqual(brier_score(probs, ["H"]), 6 / 9, places=6)

    def test_rps_respects_ordinal_closeness(self):
        # Away team wins. Predicting Draw (adjacent to Away in A<D<H order)
        # must be penalised LESS than predicting Home (two steps away).
        predict_draw = ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), ["A"])
        predict_home = ranked_probability_score(np.array([[1.0, 0.0, 0.0]]), ["A"])
        self.assertLess(predict_draw, predict_home)

    def test_rps_perfect_prediction_is_zero(self):
        self.assertAlmostEqual(
            ranked_probability_score(np.array([[1.0, 0.0, 0.0]]), ["H"]), 0.0, places=9
        )

    def test_evaluate_all_returns_all_three(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        result = evaluate_all(probs, ["H", "A"])
        self.assertEqual(set(result), {"log_loss", "brier_score", "rps"})
        self.assertTrue(all(math.isfinite(v) for v in result.values()))


if __name__ == "__main__":
    unittest.main()
