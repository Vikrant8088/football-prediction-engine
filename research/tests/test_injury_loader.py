"""Unit tests for the API-Football injury loader, the join onto matches, and
the importance-weighted absence features."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.data.injury_loader as injury_loader
import research.data.player_importance as player_importance
from research.data.injury_loader import add_injury_features, load_missing_counts

MAX_MIN = 38 * 90


def _rec(team, date, player_id, name, typ="Missing Fixture"):
    return {
        "player": {"id": player_id, "name": name, "type": typ},
        "team": {"id": 1, "name": team},
        "fixture": {"id": player_id, "date": f"{date}T19:00:00+00:00"},
    }


class TestInjuryLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)

        # Injuries for AF season 2022 (= Understat 2022-23).
        self._write_injuries("39", "2022", [
            _rec("Arsenal", "2022-08-05", 1, "S. Star"),      # ever-present last season
            _rec("Arsenal", "2022-08-05", 2, "F. Fringe"),    # barely played last season
            _rec("Wolves", "2022-08-05", 3, "S. Star"),       # team name needs mapping
            _rec("Arsenal", "2022-08-05", 9, "D. Doubt", "Questionable"),  # must NOT count
        ])
        for year in ("2023", "2024"):
            self._write_injuries("39", year, [])

        # Understat player minutes for the PREVIOUS seasons (2021/2022/2023).
        for year in ("2021", "2022", "2023"):
            self._write_players(year, [
                {"player_name": "Sammy Star", "time": str(MAX_MIN)},   # importance 1.0
                {"player_name": "Freddie Fringe", "time": str(MAX_MIN // 10)},  # 0.1
            ])

    def tearDown(self):
        self._tmp.cleanup()

    def _write_injuries(self, league, season, recs):
        base = self.raw / "api_football" / "injuries" / league / season
        (base / "v1").mkdir(parents=True)
        (base / "v1" / f"injuries_{league}_{season}.json").write_text(
            json.dumps(recs), encoding="utf-8"
        )
        (base / "latest.json").write_text(json.dumps({"latest_version": "v1"}), encoding="utf-8")

    def _write_players(self, af_season, players):
        base = self.raw / "understat" / "EPL" / af_season
        (base / "v1").mkdir(parents=True)
        (base / "v1" / f"EPL_{af_season}.json").write_text(
            json.dumps({"teams": {}, "dates": [], "players": players}), encoding="utf-8"
        )
        (base / "latest.json").write_text(json.dumps({"latest_version": "v1"}), encoding="utf-8")

    def _patched(self):
        cfg = SimpleNamespace(raw_data_dir=self.raw)
        return patch.multiple(
            injury_loader, load_config=lambda *a, **k: cfg
        ), patch.object(player_importance, "load_config", return_value=cfg)

    def _run(self, fn, *args):
        cfg = SimpleNamespace(raw_data_dir=self.raw)
        with patch.object(injury_loader, "load_config", return_value=cfg), patch.object(
            player_importance, "load_config", return_value=cfg
        ):
            return fn(*args)

    def test_counts_exclude_questionable_and_map_team_names(self):
        counts = self._run(load_missing_counts)
        arsenal = counts[counts["team"] == "Arsenal"]
        self.assertEqual(int(arsenal["missing"].iloc[0]), 2)  # Questionable excluded
        self.assertIn("Wolverhampton Wanderers", set(counts["team"]))
        self.assertNotIn("Wolves", set(counts["team"]))

    def test_weight_reflects_player_importance_not_just_count(self):
        counts = self._run(load_missing_counts)
        arsenal = counts[counts["team"] == "Arsenal"].iloc[0]
        # 2 players missing, but importance 1.0 (Star) + 0.1 (Fringe) = 1.1
        self.assertEqual(int(arsenal["missing"]), 2)
        self.assertAlmostEqual(float(arsenal["weight"]), 1.1, places=2)

    def test_features_join_onto_matches(self):
        matches = pd.DataFrame([
            {"date": pd.Timestamp("2022-08-05 19:00:00"), "season": "2022-23",
             "home_team": "Arsenal", "away_team": "Wolverhampton Wanderers"},
            {"date": pd.Timestamp("2022-08-06 15:00:00"), "season": "2022-23",
             "home_team": "Chelsea", "away_team": "Arsenal"},
        ])
        out = self._run(add_injury_features, matches)
        row0 = out.iloc[0]
        self.assertEqual(row0["home_injuries"], 2.0)
        self.assertEqual(row0["away_injuries"], 1.0)
        # Weighted: Arsenal 1.1 vs Wolves 1.0 (one Star) -> diff 0.1
        self.assertAlmostEqual(row0["home_injury_weight"], 1.1, places=2)
        self.assertAlmostEqual(row0["away_injury_weight"], 1.0, places=2)
        self.assertAlmostEqual(row0["injury_weight_diff"], 0.1, places=2)
        # No injuries that date -> genuine zeros.
        self.assertEqual(out.iloc[1]["home_injuries"], 0.0)
        self.assertEqual(out.iloc[1]["home_injury_weight"], 0.0)

    def test_uncovered_seasons_are_nan(self):
        matches = pd.DataFrame([
            {"date": pd.Timestamp("2019-08-10 15:00:00"), "season": "2019-20",
             "home_team": "Arsenal", "away_team": "Chelsea"},
        ])
        out = self._run(add_injury_features, matches)
        self.assertTrue(np.isnan(out.iloc[0]["home_injuries"]))
        self.assertTrue(np.isnan(out.iloc[0]["home_injury_weight"]))
        self.assertTrue(np.isnan(out.iloc[0]["injury_weight_diff"]))


if __name__ == "__main__":
    unittest.main()
