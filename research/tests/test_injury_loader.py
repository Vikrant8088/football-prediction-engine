"""Unit tests for the API-Football injury loader + join onto matches."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.data.injury_loader as injury_loader
from research.data.injury_loader import add_injury_features, load_missing_counts


def _rec(team, date, player_id, typ="Missing Fixture"):
    return {
        "player": {"id": player_id, "name": f"P{player_id}", "type": typ},
        "team": {"id": 1, "name": team},
        "fixture": {"id": player_id, "date": f"{date}T19:00:00+00:00"},
    }


class TestInjuryLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)
        # Write a synthetic injuries file for AF season 2022 only (=2022-23).
        recs = [
            _rec("Arsenal", "2022-08-05", 1),
            _rec("Arsenal", "2022-08-05", 2),           # Arsenal: 2 missing that day
            _rec("Wolves", "2022-08-05", 3),            # name needs mapping
            _rec("Arsenal", "2022-08-05", 9, "Questionable"),  # must NOT count
        ]
        self._write("39", "2022", recs)
        for year in ("2023", "2024"):
            self._write("39", year, [])  # empty other covered seasons

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, league, season, recs):
        d = self.raw / "api_football" / "injuries" / league / season / "v1"
        d.mkdir(parents=True)
        (d / f"injuries_{league}_{season}.json").write_text(json.dumps(recs), encoding="utf-8")
        (self.raw / "api_football" / "injuries" / league / season / "latest.json").write_text(
            json.dumps({"latest_version": "v1"}), encoding="utf-8"
        )

    def _patched(self):
        fake_cfg = SimpleNamespace(raw_data_dir=self.raw)
        return patch.object(injury_loader, "load_config", return_value=fake_cfg)

    def test_counts_only_missing_fixture_and_maps_team_names(self):
        with self._patched():
            counts = load_missing_counts()
        arsenal = counts[counts["team"] == "Arsenal"]
        self.assertEqual(int(arsenal["missing"].iloc[0]), 2)  # Questionable excluded
        # 'Wolves' mapped to the Understat canonical name.
        self.assertIn("Wolverhampton Wanderers", set(counts["team"]))
        self.assertNotIn("Wolves", set(counts["team"]))

    def test_features_join_onto_matches(self):
        matches = pd.DataFrame([
            {"date": pd.Timestamp("2022-08-05 19:00:00"), "season": "2022-23",
             "home_team": "Arsenal", "away_team": "Wolverhampton Wanderers"},
            {"date": pd.Timestamp("2022-08-06 15:00:00"), "season": "2022-23",
             "home_team": "Chelsea", "away_team": "Arsenal"},  # nobody missing that day -> 0
        ])
        with self._patched():
            out = add_injury_features(matches)
        row0 = out.iloc[0]
        self.assertEqual(row0["home_injuries"], 2.0)   # Arsenal
        self.assertEqual(row0["away_injuries"], 1.0)   # Wolves (mapped)
        self.assertEqual(row0["injuries_diff"], 1.0)
        # Second match: no injuries recorded that date -> genuine zeros.
        self.assertEqual(out.iloc[1]["home_injuries"], 0.0)
        self.assertEqual(out.iloc[1]["away_injuries"], 0.0)

    def test_uncovered_seasons_are_nan(self):
        matches = pd.DataFrame([
            {"date": pd.Timestamp("2019-08-10 15:00:00"), "season": "2019-20",
             "home_team": "Arsenal", "away_team": "Chelsea"},
        ])
        with self._patched():
            out = add_injury_features(matches)
        self.assertTrue(np.isnan(out.iloc[0]["home_injuries"]))
        self.assertTrue(np.isnan(out.iloc[0]["injuries_diff"]))


if __name__ == "__main__":
    unittest.main()
