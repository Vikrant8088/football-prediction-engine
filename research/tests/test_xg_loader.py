"""Unit tests for the Understat xG research loader.

These build a tiny synthetic raw lake on disk (the versioned layout the ingest
layer writes) and patch `load_config` so the loader reads the temp lake instead
of the real one - no network, no dependency on what has actually been
downloaded.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import research.data.xg_loader as xg_loader
from research.data.xg_loader import _season_label, load_understat_matches


def _match(home, away, hg, ag, hx, ax, dt, is_result=True):
    entry = {
        "id": f"{home}{away}{dt}",
        "isResult": is_result,
        "h": {"title": home},
        "a": {"title": away},
        "datetime": dt,
    }
    if is_result:
        entry["goals"] = {"h": str(hg), "a": str(ag)}
        entry["xG"] = {"h": str(hx), "a": str(ax)}
    return entry


class TestXgLoader(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_season(self, league, year, dates):
        dataset_dir = self.raw / "understat" / league / year
        version_dir = dataset_dir / "v1"
        version_dir.mkdir(parents=True)
        (version_dir / f"{league}_{year}.json").write_text(
            json.dumps({"teams": {}, "players": [], "dates": dates}),
            encoding="utf-8",
        )
        (dataset_dir / "latest.json").write_text(
            json.dumps({"latest_version": "v1"}), encoding="utf-8"
        )

    def _load(self, league="EPL", seasons=("2014",)):
        fake_cfg = SimpleNamespace(
            raw_data_dir=self.raw,
            understat=SimpleNamespace(seasons=list(seasons)),
        )
        with patch.object(xg_loader, "load_config", return_value=fake_cfg):
            return load_understat_matches(league)

    def test_season_label_mapping(self):
        self.assertEqual(_season_label("2014"), "2014-15")
        self.assertEqual(_season_label("2019"), "2019-20")
        self.assertEqual(_season_label("2025"), "2025-26")

    def test_parses_fields_and_types(self):
        self._write_season("EPL", "2014", [
            _match("Arsenal", "Chelsea", 2, 1, 1.9, 1.2, "2014-08-16 15:00:00"),
        ])
        df = self._load(seasons=["2014"])
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["home_team"], "Arsenal")
        self.assertEqual(row["away_team"], "Chelsea")
        self.assertEqual(row["home_goals"], 2)
        self.assertEqual(row["away_goals"], 1)
        self.assertEqual(row["season"], "2014-15")
        self.assertAlmostEqual(row["home_xg"], 1.9)
        self.assertAlmostEqual(row["away_xg"], 1.2)
        # goals/xg parsed to numeric, not left as strings
        self.assertEqual(df["home_goals"].dtype.kind, "i")
        self.assertEqual(df["home_xg"].dtype.kind, "f")

    def test_result_derived_from_goals(self):
        self._write_season("EPL", "2014", [
            _match("A", "B", 2, 0, 1.5, 0.5, "2014-08-16 15:00:00"),  # H
            _match("C", "D", 1, 1, 1.0, 1.0, "2014-08-16 17:00:00"),  # D
            _match("E", "F", 0, 3, 0.4, 2.2, "2014-08-16 19:00:00"),  # A
        ])
        df = self._load(seasons=["2014"])
        self.assertEqual(list(df["result"]), ["H", "D", "A"])

    def test_unplayed_matches_are_skipped(self):
        self._write_season("EPL", "2014", [
            _match("A", "B", 2, 0, 1.5, 0.5, "2014-08-16 15:00:00"),
            _match("C", "D", 0, 0, 0, 0, "2099-05-01 15:00:00", is_result=False),
        ])
        df = self._load(seasons=["2014"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["home_team"], "A")

    def test_multiple_seasons_concatenated_and_sorted(self):
        self._write_season("EPL", "2015", [
            _match("X", "Y", 1, 0, 1.1, 0.3, "2015-08-08 15:00:00"),
        ])
        self._write_season("EPL", "2014", [
            _match("A", "B", 2, 0, 1.5, 0.5, "2014-08-16 15:00:00"),
        ])
        df = self._load(seasons=["2014", "2015"])
        self.assertEqual(len(df), 2)
        # Chronologically sorted regardless of season load order.
        self.assertTrue(df["date"].is_monotonic_increasing)
        self.assertEqual(list(df["season"]), ["2014-15", "2015-16"])

    def test_missing_season_is_skipped_not_fatal(self):
        self._write_season("EPL", "2014", [
            _match("A", "B", 2, 0, 1.5, 0.5, "2014-08-16 15:00:00"),
        ])
        # 2015 requested but never written -> should be skipped, 2014 still loads.
        df = self._load(seasons=["2014", "2015"])
        self.assertEqual(len(df), 1)

    def test_no_data_raises(self):
        with self.assertRaises(ValueError):
            self._load(seasons=["2014"])  # nothing written at all


if __name__ == "__main__":
    unittest.main()
