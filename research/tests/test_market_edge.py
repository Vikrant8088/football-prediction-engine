"""Tests for the market-edge machinery: blend-weight fitting and CLV."""

import unittest

import numpy as np
import pandas as pd

from research.evaluation.benchmark_market_edge import closing_line_value, fit_blend_weight


class TestFitBlendWeight(unittest.TestCase):
    def test_useless_engine_gets_zero_weight(self):
        # Market is right; engine is garbage. Optimal weight on engine is 0.
        n = 200
        outcomes = ["H"] * n
        market = np.tile([0.9, 0.05, 0.05], (n, 1))
        engine = np.tile([0.05, 0.05, 0.9], (n, 1))
        self.assertAlmostEqual(fit_blend_weight(engine, market, outcomes), 0.0, places=6)

    def test_perfect_engine_gets_full_weight(self):
        n = 200
        outcomes = ["H"] * n
        market = np.tile([0.34, 0.33, 0.33], (n, 1))
        engine = np.tile([0.98, 0.01, 0.01], (n, 1))
        self.assertAlmostEqual(fit_blend_weight(engine, market, outcomes), 1.0, places=6)

    def test_complementary_forecasts_get_intermediate_weight(self):
        # Each is right on half the matches -> blending should beat either alone.
        outcomes = ["H"] * 100 + ["A"] * 100
        engine = np.vstack([np.tile([0.8, 0.1, 0.1], (100, 1)), np.tile([0.34, 0.33, 0.33], (100, 1))])
        market = np.vstack([np.tile([0.34, 0.33, 0.33], (100, 1)), np.tile([0.1, 0.1, 0.8], (100, 1))])
        w = fit_blend_weight(engine, market, outcomes)
        self.assertGreater(w, 0.05)
        self.assertLess(w, 0.95)

    def test_weight_is_bounded(self):
        n = 50
        rng = np.random.RandomState(0)
        engine = np.tile([0.4, 0.3, 0.3], (n, 1))
        market = np.tile([0.3, 0.3, 0.4], (n, 1))
        outcomes = rng.choice(["H", "D", "A"], n).tolist()
        w = fit_blend_weight(engine, market, outcomes)
        self.assertGreaterEqual(w, 0.0)
        self.assertLessEqual(w, 1.0)


class TestClosingLineValue(unittest.TestCase):
    def _frames(self, engine_home, open_home, close_home):
        n = len(engine_home)
        engine_df = pd.DataFrame({
            "season": ["2020-21"] * n,
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "p_home": engine_home,
            "p_draw": [0.25] * n,
            "p_away": [1 - h - 0.25 for h in engine_home],
        })
        merged = pd.DataFrame({
            "season": ["2020-21"] * n,
            "home_team": [f"H{i}" for i in range(n)],
            "away_team": [f"A{i}" for i in range(n)],
            "open_p_home": open_home, "open_p_draw": [0.25] * n,
            "open_p_away": [1 - h - 0.25 for h in open_home],
            "close_p_home": close_home, "close_p_draw": [0.25] * n,
            "close_p_away": [1 - h - 0.25 for h in close_home],
        })
        return engine_df, merged

    def test_line_moving_toward_us_gives_positive_correlation(self):
        engine = [0.6, 0.3, 0.55, 0.35, 0.5, 0.45]
        opening = [0.5, 0.4, 0.45, 0.45, 0.5, 0.5]
        # Closing moves halfway toward the engine each time.
        closing = [(e + o) / 2 for e, o in zip(engine, opening)]
        engine_df, merged = self._frames(engine, opening, closing)
        clv = closing_line_value(merged, engine_df)
        self.assertGreater(clv["correlation"], 0.5)
        self.assertGreater(clv["share_line_moved_toward_us"], 0.9)

    def test_line_moving_against_us_gives_negative_correlation(self):
        engine = [0.6, 0.3, 0.55, 0.35]
        opening = [0.5, 0.4, 0.45, 0.45]
        closing = [o - (e - o) for e, o in zip(engine, opening)]  # moves away
        engine_df, merged = self._frames(engine, opening, closing)
        clv = closing_line_value(merged, engine_df)
        self.assertLess(clv["correlation"], -0.5)


if __name__ == "__main__":
    unittest.main()
