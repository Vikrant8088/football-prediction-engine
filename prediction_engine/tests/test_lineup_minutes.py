"""Tests for the predicted-lineup Start % minutes model (Phase 6f candidate).

Start % is P(the manager picks him) — NOT P(60+) and NOT P(plays). The whole point of
`lineup_minutes` is to keep those three distinct, so these tests pin that separation,
the per-player cameo shape, and — most importantly — that the feed is OFF by default:
an unproven signal must not change the shipped engine until Bank-It says it earns it.
"""

import unittest

import pandas as pd

from prediction_engine.fpl.minutes import (
    MEAN_MINUTES_WHEN_CAMEO,
    MEAN_MINUTES_WHEN_LONG,
    P_60_GIVEN_START,
    lineup_minutes,
    recent_form_minutes,
)
from prediction_engine.fpl.projection import _live_minutes_model


def _player(player_id=1, chance=100.0, available=True):
    return pd.Series({"id": player_id, "chance_of_playing": chance,
                      "available": available, "minutes": 900, "position": 3})


class TestLineupMinutes(unittest.TestCase):
    def test_start_pct_is_not_treated_as_p60(self):
        # A 100%-certain starter is still only ~93.5% to last the hour.
        model = lineup_minutes(100)
        self.assertAlmostEqual(model["p_60"], P_60_GIVEN_START, places=6)
        self.assertLess(model["p_60"], 1.0)

    def test_certain_starter_plays(self):
        model = lineup_minutes(100)
        self.assertAlmostEqual(model["p_play"], 1.0, places=6)
        # Both terms matter: even a certain starter is hooked before the hour 6.5% of
        # the time, and those minutes still count.
        expected = (P_60_GIVEN_START * MEAN_MINUTES_WHEN_LONG
                    + (1.0 - P_60_GIVEN_START) * MEAN_MINUTES_WHEN_CAMEO)
        self.assertAlmostEqual(model["expected_minutes"], expected, places=2)
        self.assertTrue(78.0 < model["expected_minutes"] < 86.0,
                        "an ever-present starter should project near a full match")

    def test_benched_player_with_no_history_is_zeroed(self):
        model = lineup_minutes(0)
        self.assertEqual(model["p_60"], 0.0)
        self.assertEqual(model["p_play"], 0.0)
        self.assertEqual(model["expected_minutes"], 0.0)

    def test_non_starter_can_still_cameo_using_his_own_history(self):
        # Never starts, but always comes on: p_60 hist 0, p_play hist 1.
        recent = {"expected_minutes": 20.0, "p_60": 0.0, "p_play": 1.0}
        model = lineup_minutes(0, recent=recent)
        self.assertEqual(model["p_60"], 0.0)          # he will not start
        self.assertAlmostEqual(model["p_play"], 1.0)  # but he always features
        self.assertGreater(model["expected_minutes"], 0.0)

    def test_cameo_shape_is_per_player_not_a_constant(self):
        # Same Start %, different bench behaviour -> different p_play.
        sub = {"expected_minutes": 20.0, "p_60": 0.0, "p_play": 1.0}   # always on
        reserve = {"expected_minutes": 0.0, "p_60": 0.0, "p_play": 0.0}  # never moves
        self.assertGreater(lineup_minutes(20, recent=sub)["p_play"],
                           lineup_minutes(20, recent=reserve)["p_play"])

    def test_ordering_invariant_holds(self):
        recent = {"expected_minutes": 45.0, "p_60": 0.4, "p_play": 0.7}
        for pct in (0, 10, 50, 90, 100):
            model = lineup_minutes(pct, recent=recent)
            self.assertLessEqual(model["p_60"], model["p_play"] + 1e-9,
                                 "P(60+) can never exceed P(plays)")
            self.assertLessEqual(model["expected_minutes"], 90.0)

    def test_availability_still_applies_over_the_feed(self):
        # The feed can be stale; FPL's flag is the official word.
        full = lineup_minutes(100, availability=1.0)
        injured = lineup_minutes(100, availability=0.0)
        self.assertEqual(injured["p_60"], 0.0)
        self.assertGreater(full["p_60"], 0.0)

    def test_start_pct_monotonic_in_p60(self):
        self.assertLess(lineup_minutes(25)["p_60"], lineup_minutes(75)["p_60"])


class TestProjectionWiring(unittest.TestCase):
    """The feed must be opt-in: absent it, nothing about the engine changes."""

    HISTORY = {1: [90, 90, 90, 90]}

    def test_off_by_default_uses_recent_form(self):
        model = _live_minutes_model(_player(), self.HISTORY)
        expected = recent_form_minutes(self.HISTORY[1], half_life_matches=2.0,
                                       availability=1.0)
        self.assertEqual(model, expected)

    def test_player_absent_from_the_feed_keeps_recent_form(self):
        # A partial feed must degrade per-player, not wholesale.
        model = _live_minutes_model(_player(player_id=1), self.HISTORY, start_pct={999: 90})
        self.assertEqual(model, recent_form_minutes(self.HISTORY[1],
                                                    half_life_matches=2.0,
                                                    availability=1.0))

    def test_feed_overrides_recent_form_when_present(self):
        # An ever-present starter the feed says is benched: recent form alone would
        # keep projecting him at ~90 minutes. This is the rotation signal.
        model = _live_minutes_model(_player(), self.HISTORY, start_pct={1: 0})
        self.assertEqual(model["p_60"], 0.0)
        self.assertLess(model["expected_minutes"], 5.0)

    def test_feed_promotes_a_surprise_starter(self):
        # No recent minutes at all, but the feed says he starts: recent-form would
        # project ~0. This is exactly the case the archive exists to capture.
        model = _live_minutes_model(_player(player_id=2), {2: [0, 0, 0, 0]},
                                    start_pct={2: 90})
        self.assertGreater(model["p_60"], 0.8)
        self.assertGreater(model["expected_minutes"], 60.0)

    def test_injury_flag_still_beats_an_optimistic_feed(self):
        model = _live_minutes_model(_player(chance=0.0, available=False),
                                    self.HISTORY, start_pct={1: 95})
        self.assertEqual(model["p_60"], 0.0)


if __name__ == "__main__":
    unittest.main()
