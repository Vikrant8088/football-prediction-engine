"""Unit tests for the Understat<->FPL player join and the xG substitution.

A wrong join silently feeds one player's xG into another's projection, so the
matcher's precision matters as much as its coverage. These pin the behaviours the
matcher relies on: club disambiguation, common-vs-legal-name resolution, the
transfer fallback, and refusing an ambiguous match.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.data.understat_fpl_player_map as pmap
import research.data.understat_xg_join as xgjoin


def _fpl_meta(rows):
    # rows: list of (element, name, team, position)
    return {e: {"name": n, "team": t, "position": p} for e, n, t, p in rows}


def _us_players(rows):
    # rows: list of (uid, name, team)
    return {u: {"name": n, "team": t} for u, n, t in rows}


class TestPlayerMatching(unittest.TestCase):
    def _build(self, fpl_rows, us_rows):
        with patch.object(pmap, "load_player_meta", return_value=_fpl_meta(fpl_rows)), \
             patch.object(pmap, "understat_players", return_value=_us_players(us_rows)):
            return pmap.build_map("2020-21")

    def test_exact_name_and_team(self):
        m = self._build([(1, "Mohamed Salah", "Liverpool", 3)],
                        [("100", "Mohamed Salah", "Liverpool")])
        self.assertEqual(m, {1: "100"})

    def test_full_legal_name_matches_common_name_same_club(self):
        # FPL "Alisson Ramses Becker" vs Understat "Alisson".
        m = self._build([(1, "Alisson Ramses Becker", "Liverpool", 1)],
                        [("100", "Alisson", "Liverpool")])
        self.assertEqual(m, {1: "100"})

    def test_abbreviated_forename_same_surname(self):
        # "Benjamin Chilwell" vs "Ben Chilwell".
        m = self._build([(1, "Benjamin Chilwell", "Chelsea", 2)],
                        [("100", "Ben Chilwell", "Chelsea")])
        self.assertEqual(m, {1: "100"})

    def test_hyphenated_surname_tokenises(self):
        # "Ward-Prowse" must become ["ward","prowse"], not "wardprowse".
        m = self._build([(1, "James Ward-Prowse", "Southampton", 3)],
                        [("100", "James Ward-Prowse", "Southampton")])
        self.assertEqual(m, {1: "100"})

    def test_nordic_letter_is_transliterated(self):
        # NFKD drops o-slash; without the map "Odegaard" and "Ødegaard" never meet.
        m = self._build([(1, "Martin Ødegaard", "Arsenal", 3)],
                        [("100", "Martin Odegaard", "Arsenal")])
        self.assertEqual(m, {1: "100"})

    def test_alias_resolves_a_nickname(self):
        m = self._build([(1, "Fabio Henrique Tavares", "Liverpool", 3)],
                        [("100", "Fabinho", "Liverpool")])
        self.assertEqual(m, {1: "100"})

    def test_same_surname_different_club_does_not_match(self):
        # The club safeguard: two "Reid"s at different clubs must not be confused.
        m = self._build([(1, "Winston Reid", "West Ham", 2)],
                        [("100", "Steven Reid", "Fulham")])
        self.assertEqual(m, {})

    def test_two_same_surname_teammates_need_the_forename(self):
        # Both Chelsea; only the forename separates them.
        m = self._build(
            [(1, "Reece James", "Chelsea", 2), (2, "Daniel James", "Chelsea", 3)],
            [("100", "Reece James", "Chelsea"), ("200", "Dan James", "Chelsea")])
        self.assertEqual(m[1], "100")
        self.assertEqual(m[2], "200")

    def test_transfer_fallback_matches_across_clubs_by_exact_name(self):
        # FPL has him at his new club; Understat aggregates him at the old one.
        m = self._build([(1, "Cole Palmer", "Chelsea", 3)],
                        [("100", "Cole Palmer", "Manchester City")])
        self.assertEqual(m, {1: "100"})

    def test_transfer_fallback_requires_a_unique_exact_name(self):
        # Two exact-name Understat players and no club match -> refuse, don't guess.
        m = self._build([(1, "Danny Ward", "Leicester", 1)],
                        [("100", "Danny Ward", "Nottingham Forest"),
                         ("200", "Danny Ward", "Huddersfield")])
        self.assertEqual(m, {})

    def test_an_understat_id_is_claimed_once(self):
        m = self._build(
            [(1, "Joe Gomez", "Liverpool", 2), (2, "Joel Matip", "Liverpool", 2)],
            [("100", "Joe Gomez", "Liverpool")])
        self.assertEqual(list(m.values()).count("100"), 1)


class TestXgInjection(unittest.TestCase):
    def _frame(self):
        # Two players; player 1 has a double gameweek (two fixtures, two dates).
        return pd.DataFrame([
            {"player_id": 1, "kickoff_time": pd.Timestamp("2020-09-12", tz="UTC"),
             "minutes": 90, "expected_goals": 9.9, "expected_assists": 9.9},
            {"player_id": 1, "kickoff_time": pd.Timestamp("2020-09-15", tz="UTC"),
             "minutes": 90, "expected_goals": 9.9, "expected_assists": 9.9},
            {"player_id": 2, "kickoff_time": pd.Timestamp("2020-09-12", tz="UTC"),
             "minutes": 0, "expected_goals": 9.9, "expected_assists": 9.9},
            {"player_id": 3, "kickoff_time": pd.Timestamp("2020-09-12", tz="UTC"),
             "minutes": 90, "expected_goals": 9.9, "expected_assists": 9.9},
        ])

    def _run(self):
        mapping = {1: "u1", 2: "u2"}          # player 3 is unmatched
        matches = {
            "u1": [
                {"season": "2020", "date": "2020-09-12", "xG": "0.5", "xA": "0.1"},
                {"season": "2020", "date": "2020-09-15", "xG": "0.7", "xA": "0.2"},
                {"season": "2019", "date": "2019-09-12", "xG": "9.0", "xA": "9.0"},
            ],
            "u2": [{"season": "2020", "date": "2020-09-12", "xG": "0.3", "xA": "0.0"}],
        }
        with patch.object(xgjoin, "build_map", return_value=mapping):
            return xgjoin.inject_understat_xg(self._frame(), "2020-21", matches=matches)

    def test_each_fixture_gets_its_own_days_xg(self):
        frame, _ = self._run()
        player1 = frame[frame["player_id"] == 1].sort_values("kickoff_time")
        self.assertAlmostEqual(player1.iloc[0]["expected_goals"], 0.5)   # 12th
        self.assertAlmostEqual(player1.iloc[1]["expected_goals"], 0.7)   # 15th

    def test_only_this_seasons_matches_are_used(self):
        frame, _ = self._run()
        # The 2019 match (xG 9.0) must never leak into a 2020-21 row.
        self.assertNotIn(9.0, list(frame["expected_goals"]))

    def test_unmatched_player_gets_zero_not_the_old_value(self):
        frame, _ = self._run()
        player3 = frame[frame["player_id"] == 3].iloc[0]
        self.assertEqual(player3["expected_goals"], 0.0)   # not the placeholder 9.9

    def test_coverage_counts_played_minutes_only(self):
        _, summary = self._run()
        # Players 1 (two 90s) and 2 (0 min) mapped, player 3 (90) not. Player 2's row
        # has no minutes so it does not count against coverage.
        self.assertAlmostEqual(summary["minute_coverage"], 180 / 270)   # 1 covered, 3 not
        self.assertEqual(summary["mapped_players"], 2)


if __name__ == "__main__":
    unittest.main()
