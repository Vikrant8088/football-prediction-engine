"""Tests for the FPL scoring rules.

These are checked against real scored matches in
`research/evaluation/validate_fpl_scoring.py` (2,085/2,085 exact). These unit
tests pin the individual rules so a future edit cannot break one silently.
"""

import unittest

from prediction_engine.fpl.scoring import (
    DEF,
    FWD,
    GKP,
    MID,
    appearance_points,
    defensive_contribution_points,
    match_points,
    match_points_from_history,
)


class TestAppearance(unittest.TestCase):
    def test_no_minutes_scores_nothing(self):
        self.assertEqual(appearance_points(0), 0)
        self.assertEqual(match_points(MID, minutes=0, goals_scored=3), 0)

    def test_short_and_long_appearances(self):
        self.assertEqual(appearance_points(1), 1)
        self.assertEqual(appearance_points(59), 1)
        self.assertEqual(appearance_points(60), 2)  # boundary is inclusive
        self.assertEqual(appearance_points(90), 2)


class TestGoalsAndAssists(unittest.TestCase):
    def test_goal_values_by_position(self):
        for position, expected in [(GKP, 6), (DEF, 6), (MID, 5), (FWD, 4)]:
            pts = match_points(position, minutes=90, goals_scored=1)
            self.assertEqual(pts, 2 + expected, msg=f"position {position}")

    def test_assists_are_three_for_everyone(self):
        for position in (GKP, DEF, MID, FWD):
            self.assertEqual(match_points(position, minutes=90, assists=1), 2 + 3)


class TestCleanSheetsAndConceding(unittest.TestCase):
    def test_clean_sheet_values(self):
        self.assertEqual(match_points(GKP, minutes=90, clean_sheets=1), 2 + 4)
        self.assertEqual(match_points(DEF, minutes=90, clean_sheets=1), 2 + 4)
        self.assertEqual(match_points(MID, minutes=90, clean_sheets=1), 2 + 1)
        self.assertEqual(match_points(FWD, minutes=90, clean_sheets=1), 2 + 0)

    def test_conceded_penalty_only_for_gk_and_def(self):
        # 3 conceded -> floor(3/2) = 1 point deducted
        self.assertEqual(match_points(GKP, minutes=90, goals_conceded=3), 2 - 1)
        self.assertEqual(match_points(DEF, minutes=90, goals_conceded=3), 2 - 1)
        self.assertEqual(match_points(MID, minutes=90, goals_conceded=3), 2)
        self.assertEqual(match_points(FWD, minutes=90, goals_conceded=3), 2)

    def test_one_conceded_costs_nothing(self):
        self.assertEqual(match_points(DEF, minutes=90, goals_conceded=1), 2)


class TestGoalkeeperSpecials(unittest.TestCase):
    def test_saves_are_one_point_per_three(self):
        self.assertEqual(match_points(GKP, minutes=90, saves=2), 2 + 0)
        self.assertEqual(match_points(GKP, minutes=90, saves=3), 2 + 1)
        self.assertEqual(match_points(GKP, minutes=90, saves=7), 2 + 2)

    def test_outfield_saves_score_nothing(self):
        self.assertEqual(match_points(DEF, minutes=90, saves=6), 2)

    def test_penalty_save(self):
        self.assertEqual(match_points(GKP, minutes=90, penalties_saved=1), 2 + 5)


class TestDefensiveContribution(unittest.TestCase):
    def test_thresholds_by_position(self):
        self.assertEqual(defensive_contribution_points(DEF, 9), 0)
        self.assertEqual(defensive_contribution_points(DEF, 10), 2)  # DEF threshold
        self.assertEqual(defensive_contribution_points(MID, 11), 0)
        self.assertEqual(defensive_contribution_points(MID, 12), 2)  # MID/FWD threshold
        self.assertEqual(defensive_contribution_points(FWD, 12), 2)

    def test_goalkeepers_are_not_eligible(self):
        self.assertEqual(defensive_contribution_points(GKP, 50), 0)


class TestPenaltiesAndCards(unittest.TestCase):
    def test_cards_and_own_goals(self):
        self.assertEqual(match_points(MID, minutes=90, yellow_cards=1), 2 - 1)
        self.assertEqual(match_points(MID, minutes=90, red_cards=1), 2 - 3)
        self.assertEqual(match_points(MID, minutes=90, own_goals=1), 2 - 2)
        self.assertEqual(match_points(FWD, minutes=90, penalties_missed=1), 2 - 2)


class TestRealMatches(unittest.TestCase):
    """Real rows taken from the FPL API, reproduced exactly."""

    def test_gabriel_clean_sheet_match(self):
        # 90 min (2) + clean sheet (4); DC 7 < 10 so no defensive points.
        row = {
            "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 1,
            "goals_conceded": 0, "saves": 0, "bonus": 0, "yellow_cards": 0,
            "red_cards": 0, "own_goals": 0, "penalties_missed": 0,
            "penalties_saved": 0, "defensive_contribution": 7,
        }
        self.assertEqual(match_points_from_history(DEF, row), 6)

    def test_gabriel_conceded_one_match(self):
        # 90 min (2), no clean sheet, 1 conceded costs nothing.
        row = {
            "minutes": 90, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
            "goals_conceded": 1, "saves": 0, "bonus": 0, "yellow_cards": 0,
            "red_cards": 0, "own_goals": 0, "penalties_missed": 0,
            "penalties_saved": 0, "defensive_contribution": 7,
        }
        self.assertEqual(match_points_from_history(DEF, row), 2)

    def test_striker_brace_with_bonus(self):
        # 90 (2) + 2 goals (8) + 3 bonus = 13
        row = {"minutes": 90, "goals_scored": 2, "assists": 0, "clean_sheets": 0,
               "goals_conceded": 1, "saves": 0, "bonus": 3, "yellow_cards": 0,
               "red_cards": 0, "own_goals": 0, "penalties_missed": 0,
               "penalties_saved": 0, "defensive_contribution": 1}
        self.assertEqual(match_points_from_history(FWD, row), 13)


class TestValidation(unittest.TestCase):
    def test_unknown_position_raises(self):
        with self.assertRaises(ValueError):
            match_points(99, minutes=90)


if __name__ == "__main__":
    unittest.main()
