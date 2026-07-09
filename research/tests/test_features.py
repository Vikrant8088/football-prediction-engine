"""Tests for the point-in-time feature builders.

The headline test is `test_no_leakage_from_future_matches`: adding more (later)
matches to the dataset must NOT change the features already computed for
earlier matches. If it does, a feature is peeking at the future and every
backtest built on it is invalid. The rest check the arithmetic of rest,
congestion, form, and cold-start handling on tiny hand-built fixtures.
"""

import unittest

import numpy as np
import pandas as pd

from research.features.builders import ALL_FEATURES, add_match_features


def _match(date, home, away, hg, ag):
    return {
        "date": pd.Timestamp(date),
        "home_team": home, "away_team": away,
        "home_goals": hg, "away_goals": ag,
    }


class TestFeatureBuilders(unittest.TestCase):
    def setUp(self):
        # A small fixture list with known dates so rest/congestion are checkable.
        self.matches = pd.DataFrame([
            _match("2020-08-01", "A", "B", 2, 0),  # A win
            _match("2020-08-05", "C", "A", 1, 1),  # A draw, 4 days after A's last
            _match("2020-08-09", "A", "C", 3, 1),  # A win, 4 days later
            _match("2020-08-30", "B", "A", 0, 2),  # A win, 21 days later
        ])

    def test_no_leakage_from_future_matches(self):
        full = add_match_features(self.matches)
        # Recompute using only the first k matches; the features for those k
        # must be byte-for-byte identical (they may only depend on the past).
        for k in range(1, len(self.matches) + 1):
            truncated = add_match_features(self.matches.iloc[:k])
            a = full.iloc[:k][ALL_FEATURES].to_numpy()
            b = truncated[ALL_FEATURES].to_numpy()
            self.assertTrue(
                np.allclose(a, b, equal_nan=True),
                msg=f"features for first {k} matches changed when future matches were added",
            )

    def test_cold_start_first_match_has_no_history(self):
        out = add_match_features(self.matches)
        first = out.iloc[0]
        # A and B both make their debut here: no form, no rest, zero congestion.
        self.assertTrue(np.isnan(first["home_form_ppg"]))
        self.assertTrue(np.isnan(first["home_rest_days"]))
        self.assertEqual(first["home_congestion"], 0.0)

    def test_rest_days(self):
        out = add_match_features(self.matches)
        # 2nd match: A (away) last played 2020-08-01, this is 2020-08-05 -> 4.
        self.assertEqual(out.iloc[1]["away_rest_days"], 4)
        # 4th match: A (away) last played 2020-08-09 -> 2020-08-30 = 21 days.
        self.assertEqual(out.iloc[3]["away_rest_days"], 21)

    def test_congestion_counts_recent_matches_only(self):
        out = add_match_features(self.matches)
        # 3rd match (2020-08-09), team A: prior A matches on 08-01 and 08-05,
        # both within 14 days -> congestion 2.
        self.assertEqual(out.iloc[2]["home_congestion"], 2)
        # 4th match (2020-08-30), team A: prior A matches 08-01/08-05/08-09 are
        # all >14 days before 08-30 -> congestion 0.
        self.assertEqual(out.iloc[3]["away_congestion"], 0)

    def test_form_uses_only_past_results(self):
        out = add_match_features(self.matches)
        # 3rd match, team A: prior results are W (3 pts) then D (1 pt) -> ppg 2.0.
        self.assertAlmostEqual(out.iloc[2]["home_form_ppg"], 2.0)
        # A's goal differences: +2 then 0 -> mean +1.0.
        self.assertAlmostEqual(out.iloc[2]["home_gd_form"], 1.0)

    def test_output_preserves_original_columns_and_length(self):
        out = add_match_features(self.matches)
        self.assertEqual(len(out), len(self.matches))
        for col in ("date", "home_team", "away_team", "home_goals", "away_goals"):
            self.assertIn(col, out.columns)


if __name__ == "__main__":
    unittest.main()
