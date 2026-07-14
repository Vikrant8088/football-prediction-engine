"""Tests for the expected-minutes models.

The crude model must reproduce the shipped behaviour exactly (so switching the
projection onto this interface is a no-op until a better model is chosen); the
recent-form model must read the rotation/injury signal the flat average hides.
"""

import unittest

from prediction_engine.fpl.minutes import (
    FULL_MATCH_MINUTES,
    crude_minutes,
    recent_form_minutes,
)


class TestCrudeMinutes(unittest.TestCase):
    def test_ever_present_is_full_and_certain(self):
        m = crude_minutes(total_minutes=90 * 10, matches=10)
        self.assertAlmostEqual(m["expected_minutes"], 90.0)
        self.assertAlmostEqual(m["p_60"], 1.0)
        # The shipped model cannot tell p_60 from p_play - they are equal by design.
        self.assertEqual(m["p_60"], m["p_play"])

    def test_reproduces_shipped_p60_equals_average_over_60(self):
        # 450 minutes over 10 matches -> 45 avg -> p_60 = 45/60 = 0.75.
        m = crude_minutes(total_minutes=450, matches=10)
        self.assertAlmostEqual(m["expected_minutes"], 45.0)
        self.assertAlmostEqual(m["p_60"], 0.75)
        self.assertEqual(m["p_60"], m["p_play"])

    def test_availability_scales_down(self):
        m = crude_minutes(total_minutes=90 * 10, matches=10, availability=0.5)
        self.assertAlmostEqual(m["expected_minutes"], 45.0)

    def test_no_matches_never_divides_by_zero(self):
        m = crude_minutes(total_minutes=0, matches=0)
        self.assertEqual(m["expected_minutes"], 0.0)


class TestRecentFormMinutes(unittest.TestCase):
    def test_empty_history_returns_none(self):
        self.assertIsNone(recent_form_minutes([]))

    def test_nailed_starter(self):
        m = recent_form_minutes([90] * 8)
        self.assertAlmostEqual(m["expected_minutes"], 90.0)
        self.assertAlmostEqual(m["p_60"], 1.0)
        self.assertAlmostEqual(m["p_play"], 1.0)

    def test_permanent_substitute_is_not_clean_sheet_eligible(self):
        # Always plays, never 60+. The crude average (20) would wrongly imply a
        # 0.33 clean-sheet eligibility; the recent model correctly zeroes it.
        m = recent_form_minutes([20] * 8)
        self.assertAlmostEqual(m["p_60"], 0.0)
        self.assertAlmostEqual(m["p_play"], 1.0)
        self.assertGreater(m["expected_minutes"], 0.0)

    def test_recency_dominates_a_role_change(self):
        # Lost his place: started the season, benched lately. Recent form should
        # read much lower than the flat average of this sequence (~50).
        seq = [90, 90, 90, 90, 0, 0, 0, 0]
        m = recent_form_minutes(seq, half_life_matches=3.0)
        flat_average = sum(seq) / len(seq)
        self.assertLess(m["expected_minutes"], flat_average)
        self.assertLess(m["p_60"], 0.5)

    def test_recently_promoted_player_reads_high(self):
        # Slow start, now nailed. Recent form should read well above the flat
        # average (~50) - the opposite failure the crude model makes.
        seq = [0, 0, 0, 0, 90, 90, 90, 90]
        m = recent_form_minutes(seq, half_life_matches=3.0)
        flat_average = sum(seq) / len(seq)
        self.assertGreater(m["expected_minutes"], flat_average)
        self.assertGreater(m["p_60"], 0.5)

    def test_shorter_half_life_reacts_faster(self):
        seq = [90, 90, 90, 90, 0, 0]
        fast = recent_form_minutes(seq, half_life_matches=1.0)
        slow = recent_form_minutes(seq, half_life_matches=8.0)
        # After two benchings, the fast-decay model should sit lower.
        self.assertLess(fast["expected_minutes"], slow["expected_minutes"])

    def test_outputs_are_bounded(self):
        m = recent_form_minutes([90, 45, 0, 90, 12], half_life_matches=4.0)
        self.assertGreaterEqual(m["expected_minutes"], 0.0)
        self.assertLessEqual(m["expected_minutes"], FULL_MATCH_MINUTES)
        for key in ("p_60", "p_play"):
            self.assertGreaterEqual(m[key], 0.0)
            self.assertLessEqual(m[key], 1.0)
        # A 60+ match is always also a played match.
        self.assertGreaterEqual(m["p_play"], m["p_60"])


if __name__ == "__main__":
    unittest.main()
