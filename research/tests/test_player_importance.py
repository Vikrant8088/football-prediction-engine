"""Unit tests for player importance (previous-season minutes)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import research.data.player_importance as pi
from research.data.player_importance import (
    MAX_SEASON_MINUTES,
    ImportanceLookup,
    build_importance_lookup,
    name_key,
)


class TestNameKey(unittest.TestCase):
    def test_abbreviated_and_full_names_agree(self):
        self.assertEqual(name_key("David de Gea"), name_key("D. de Gea"))
        self.assertEqual(name_key("Virgil van Dijk"), name_key("V. van Dijk"))
        self.assertEqual(name_key("Mohamed Salah"), name_key("M. Salah"))

    def test_accents_are_stripped(self):
        self.assertEqual(name_key("Rúben Dias"), ("r", "dias"))
        self.assertEqual(name_key("R. Dias"), ("r", "dias"))

    def test_single_token_name(self):
        self.assertEqual(name_key("Fabinho"), ("", "fabinho"))

    def test_empty_name(self):
        self.assertIsNone(name_key("---"))


class TestImportanceLookup(unittest.TestCase):
    def test_exact_key_hit(self):
        lookup = ImportanceLookup({("m", "salah"): MAX_SEASON_MINUTES}, {"salah": [MAX_SEASON_MINUTES]})
        self.assertAlmostEqual(lookup.importance("M. Salah"), 1.0)

    def test_unmatched_player_scores_zero(self):
        lookup = ImportanceLookup({}, {})
        self.assertEqual(lookup.importance("A. Newsigning"), 0.0)

    def test_unique_surname_fallback(self):
        # Key ('x','kane') misses, but 'kane' is unambiguous -> use it.
        lookup = ImportanceLookup({("h", "kane"): 1710}, {"kane": [1710]})
        self.assertAlmostEqual(lookup.importance("X. Kane"), 0.5)

    def test_ambiguous_surname_refuses_to_guess(self):
        lookup = ImportanceLookup({}, {"silva": [3000, 1000]})
        self.assertEqual(lookup.importance("Z. Silva"), 0.0)

    def test_importance_is_capped_at_one(self):
        lookup = ImportanceLookup({("a", "b"): MAX_SEASON_MINUTES * 3}, {"b": [1]})
        self.assertAlmostEqual(lookup.importance("A. B"), 1.0)


class TestBuildImportanceLookup(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_season(self, af_season, players):
        d = self.raw / "understat" / "EPL" / af_season / "v1"
        d.mkdir(parents=True)
        (d / f"EPL_{af_season}.json").write_text(
            json.dumps({"teams": {}, "dates": [], "players": players}), encoding="utf-8"
        )
        (self.raw / "understat" / "EPL" / af_season / "latest.json").write_text(
            json.dumps({"latest_version": "v1"}), encoding="utf-8"
        )

    def test_builds_from_the_named_season_only(self):
        """Leakage guard: the lookup reflects the season it was asked for, and
        never the following one (callers pass the PREVIOUS season)."""
        self._write_season("2021", [{"player_name": "Star Player", "time": "3420"}])
        self._write_season("2022", [{"player_name": "Star Player", "time": "0"}])

        with patch.object(pi, "load_config", return_value=SimpleNamespace(raw_data_dir=self.raw)):
            prev = build_importance_lookup("2021")
            curr = build_importance_lookup("2022")

        self.assertAlmostEqual(prev.importance("S. Player"), 1.0)
        self.assertAlmostEqual(curr.importance("S. Player"), 0.0)

    def test_missing_season_raises(self):
        with patch.object(pi, "load_config", return_value=SimpleNamespace(raw_data_dir=self.raw)):
            with self.assertRaises(ValueError):
                build_importance_lookup("1999")


if __name__ == "__main__":
    unittest.main()
