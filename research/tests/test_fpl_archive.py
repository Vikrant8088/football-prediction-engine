"""Unit tests for the historical FPL archive loader. No network."""

import unittest
from unittest.mock import patch

import research.data.fpl_archive as fpl_archive
from research.data.fpl_archive import SEASONS_WITH_XG, load_gameweeks, load_team_names

TEAMS_CSV = b"""code,id,name,short_name
3,1,Arsenal,ARS
43,2,Man City,MCI
6,3,Spurs,TOT
49,4,Sheffield Utd,SHU
"""

# One double gameweek (Saka plays GW1 twice), one manager row, one goalkeeper.
MERGED_CSV = b"""name,position,team,element,expected_assists,expected_goals,kickoff_time,minutes,opponent_team,saves,bonus,yellow_cards,red_cards,clean_sheets,goals_conceded,goals_scored,assists,own_goals,penalties_missed,penalties_saved,total_points,value,was_home,GW
Saka,MID,Arsenal,1,0.4,0.6,2022-08-05T19:00:00Z,90,2,0,3,0,0,1,0,1,0,0,0,0,9,80,True,1
Saka,MID,Arsenal,1,0.1,0.2,2022-08-08T19:00:00Z,45,3,0,0,1,0,0,2,0,0,0,0,0,1,80,False,1
Raya,GK,Arsenal,2,0.0,0.0,2022-08-05T19:00:00Z,90,2,4,0,0,0,1,0,0,0,0,0,0,6,50,True,1
Arteta,AM,Arsenal,3,0.0,0.0,2022-08-05T19:00:00Z,0,2,0,0,0,0,0,0,0,0,0,0,0,6,15,True,1
Haaland,FWD,Man City,4,0.2,1.1,2022-08-05T19:00:00Z,90,1,0,3,0,0,0,1,2,0,0,0,0,13,120,False,1
Saka,MID,Arsenal,1,0.3,0.5,2022-08-13T19:00:00Z,90,4,0,0,0,0,1,0,0,1,0,0,0,6,81,True,2
"""


class TestLoadTeamNames(unittest.TestCase):
    def test_maps_fpl_names_to_engine_names(self):
        with patch.object(fpl_archive, "_ensure_dataset", return_value=TEAMS_CSV):
            teams = load_team_names("2022-23")
        self.assertEqual(teams[1], "Arsenal")
        self.assertEqual(teams[2], "Manchester City")
        self.assertEqual(teams[3], "Tottenham")
        # Would previously have been silently dropped from the join.
        self.assertEqual(teams[4], "Sheffield United")


class TestLoadGameweeks(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(
            fpl_archive, "_ensure_dataset",
            side_effect=lambda season, dataset, force=False: (
                TEAMS_CSV if dataset == "teams" else MERGED_CSV
            ),
        )
        self.addCleanup(patcher.stop)
        patcher.start()
        self.frame = load_gameweeks("2022-23")

    def test_rejects_seasons_without_xg(self):
        # Replaying 2016/17 on realised goals would test a different model.
        with self.assertRaises(ValueError):
            load_gameweeks("2016-17")
        self.assertNotIn("2016-17", SEASONS_WITH_XG)

    def test_manager_rows_are_dropped(self):
        self.assertNotIn("Arteta", set(self.frame["player"]))
        self.assertEqual(len(self.frame), 5)

    def test_positions_become_element_type_codes(self):
        by_name = self.frame.groupby("player")["position"].first()
        self.assertEqual(by_name["Raya"], 1)      # GK
        self.assertEqual(by_name["Saka"], 3)      # MID
        self.assertEqual(by_name["Haaland"], 4)   # FWD

    def test_opponent_id_resolves_to_canonical_name(self):
        saka_gw1 = self.frame[(self.frame["player"] == "Saka") & (self.frame["gameweek"] == 1)]
        self.assertEqual(set(saka_gw1["opponent"]), {"Manchester City", "Tottenham"})
        self.assertEqual(set(saka_gw1["team"]), {"Arsenal"})

    def test_double_gameweek_kept_as_two_fixture_rows(self):
        # The loader must NOT collapse them; the backtest decides how to combine.
        saka_gw1 = self.frame[(self.frame["player"] == "Saka") & (self.frame["gameweek"] == 1)]
        self.assertEqual(len(saka_gw1), 2)
        self.assertEqual(saka_gw1["total_points"].sum(), 10)

    def test_price_is_point_in_time(self):
        saka = self.frame[self.frame["player"] == "Saka"].sort_values("gameweek")
        self.assertEqual(list(saka["price"]), [8.0, 8.0, 8.1])

    def test_missing_defensive_contribution_column_becomes_zero(self):
        # The rule did not exist before 2025/26.
        self.assertIn("defensive_contribution", self.frame.columns)
        self.assertEqual(self.frame["defensive_contribution"].sum(), 0)

    def test_team_and_opponent_never_equal(self):
        self.assertFalse((self.frame["team"] == self.frame["opponent"]).any())

    def test_unmappable_opponent_id_raises(self):
        bad = MERGED_CSV.replace(b",2,0,3,0,0,1,0,1,0,0,0,0,9,80,True,1",
                                 b",99,0,3,0,0,1,0,1,0,0,0,0,9,80,True,1")
        with patch.object(
            fpl_archive, "_ensure_dataset",
            side_effect=lambda season, dataset, force=False: (
                TEAMS_CSV if dataset == "teams" else bad),
        ):
            with self.assertRaises(ValueError):
                load_gameweeks("2022-23")

    def test_unrecognised_position_raises(self):
        bad = MERGED_CSV.replace(b"Raya,GK,", b"Raya,WTF,")
        with patch.object(
            fpl_archive, "_ensure_dataset",
            side_effect=lambda season, dataset, force=False: (
                TEAMS_CSV if dataset == "teams" else bad),
        ):
            with self.assertRaises(ValueError):
                load_gameweeks("2022-23")


if __name__ == "__main__":
    unittest.main()
