"""Unit tests for closing-odds loading and overround removal."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.data.odds_loader as odds_loader
from research.data.odds_loader import (
    _season_label,
    implied_probabilities,
    load_closing_odds,
    overround,
)


class TestImpliedProbabilities(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        odds = np.array([[2.0, 3.5, 4.0], [1.5, 4.0, 7.0]])
        probs = implied_probabilities(odds)
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue((probs > 0).all())

    def test_fair_odds_recover_exact_probabilities(self):
        # A book with zero margin: 1/2 + 1/4 + 1/4 = 1.0 exactly.
        odds = np.array([[2.0, 4.0, 4.0]])
        probs = implied_probabilities(odds)
        self.assertTrue(np.allclose(probs[0], [0.5, 0.25, 0.25]))
        self.assertAlmostEqual(overround(odds)[0], 0.0, places=9)

    def test_overround_is_measured(self):
        # 1/2 + 1/3 + 1/6 = 1.0; shorten every price by 5% -> positive margin.
        odds = np.array([[2.0, 3.0, 6.0]]) / 1.05
        self.assertGreater(overround(odds)[0], 0.04)

    def test_shorter_odds_mean_higher_probability(self):
        probs = implied_probabilities(np.array([[1.5, 4.0, 7.0]]))[0]
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_season_label(self):
        self.assertEqual(_season_label("1819"), "2018-19")
        self.assertEqual(_season_label("2425"), "2024-25")


class TestLoadClosingOdds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_season(self, season, rows, with_odds=True):
        d = self.raw / "football_data_co_uk" / "E0" / season / "v1"
        d.mkdir(parents=True)
        df = pd.DataFrame(rows)
        if not with_odds:
            df = df.drop(columns=["PSCH", "PSCD", "PSCA"])
        df.to_csv(d / f"{season}.csv", index=False)
        (self.raw / "football_data_co_uk" / "E0" / season / "latest.json").write_text(
            '{"latest_version": "v1"}', encoding="utf-8"
        )

    def _config(self, seasons):
        return SimpleNamespace(
            raw_data_dir=self.raw,
            football_data_co_uk=SimpleNamespace(seasons=seasons),
        )

    def _load(self, seasons, map_names=True):
        with patch.object(odds_loader, "load_config", return_value=self._config(seasons)):
            return load_closing_odds("E0", map_names=map_names)

    def test_maps_team_names_and_builds_probabilities(self):
        self._write_season("1819", [
            {"Date": "11/08/2018", "HomeTeam": "Man City", "AwayTeam": "Wolves",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
            {"Date": "12/08/2018", "HomeTeam": "QPR", "AwayTeam": "Nott'm Forest",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
        ])
        out = self._load(["1819"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[0]["home_team"], "Manchester City")
        self.assertEqual(out.iloc[0]["away_team"], "Wolverhampton Wanderers")
        self.assertEqual(out.iloc[1]["home_team"], "Queens Park Rangers")
        self.assertEqual(out.iloc[1]["away_team"], "Nottingham Forest")
        self.assertEqual(out.iloc[0]["season"], "2018-19")
        self.assertAlmostEqual(out.iloc[0]["p_home"], 0.5)
        # Dates are parsed dayfirst, as football-data.co.uk publishes them.
        self.assertEqual(pd.Timestamp(out.iloc[0]["date"]), pd.Timestamp("2018-08-11"))

    def test_date_disambiguates_repeat_home_fixtures(self):
        """Regression: in leagues that split mid-season (Scottish Premiership) a
        team can host the SAME opponent twice, so (season, home, away) is not a
        unique key - the date must be part of it."""
        self._write_season("1819", [
            {"Date": "11/08/2018", "HomeTeam": "Celtic", "AwayTeam": "Rangers",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
            {"Date": "02/03/2019", "HomeTeam": "Celtic", "AwayTeam": "Rangers",
             "PSCH": 1.5, "PSCD": 4.0, "PSCA": 7.0},
        ])
        out = self._load(["1819"], map_names=False)
        self.assertEqual(len(out), 2)
        keys = set(zip(out["date"], out["home_team"], out["away_team"]))
        self.assertEqual(len(keys), 2)  # distinct once the date is included

    def test_season_without_odds_columns_is_skipped(self):
        self._write_season("1819", [
            {"Date": "11/08/2018", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
        ])
        self._write_season("2526", [
            {"Date": "11/08/2025", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
        ], with_odds=False)
        out = self._load(["1819", "2526"])
        self.assertEqual(set(out["season"]), {"2018-19"})

    def test_rows_with_missing_odds_are_dropped(self):
        self._write_season("1819", [
            {"Date": "11/08/2018", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
             "PSCH": 2.0, "PSCD": 4.0, "PSCA": 4.0},
        ] * 20 + [
            {"Date": "12/08/2018", "HomeTeam": "Everton", "AwayTeam": "Fulham",
             "PSCH": np.nan, "PSCD": 4.0, "PSCA": 4.0},
        ])
        out = self._load(["1819"])
        self.assertEqual(len(out), 20)  # the NaN-odds row is gone

    def test_no_usable_seasons_raises(self):
        with self.assertRaises(ValueError):
            self._load(["1819"])  # nothing written


class TestLoadTotalsOdds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, season, rows):
        d = self.raw / "football_data_co_uk" / "E0" / season / "v1"
        d.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(d / f"{season}.csv", index=False)
        (self.raw / "football_data_co_uk" / "E0" / season / "latest.json").write_text(
            '{"latest_version": "v1"}', encoding="utf-8"
        )

    def _load(self, seasons):
        cfg = SimpleNamespace(
            raw_data_dir=self.raw,
            football_data_co_uk=SimpleNamespace(seasons=seasons),
        )
        with patch.object(odds_loader, "load_config", return_value=cfg):
            return odds_loader.load_totals_odds("E0")

    def test_two_way_overround_is_removed(self):
        # A fair 2-way book: 1/2.0 + 1/2.0 = 1.0 exactly.
        self._write("1920", [{
            "Date": "11/08/2019", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "PC>2.5": 2.0, "PC<2.5": 2.0, "MaxC>2.5": 2.1, "MaxC<2.5": 2.05,
        }])
        out = self._load(["1920"])
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["p_over"], 0.5)
        self.assertAlmostEqual(out.iloc[0]["p_under"], 0.5)
        self.assertAlmostEqual(out.iloc[0]["p_over"] + out.iloc[0]["p_under"], 1.0)
        self.assertAlmostEqual(out.iloc[0]["overround"], 0.0, places=9)

    def test_short_over_price_means_over_is_likely(self):
        self._write("1920", [{
            "Date": "11/08/2019", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "PC>2.5": 1.4, "PC<2.5": 3.0, "MaxC>2.5": 1.45, "MaxC<2.5": 3.1,
        }])
        out = self._load(["1920"])
        self.assertGreater(out.iloc[0]["p_over"], out.iloc[0]["p_under"])

    def test_best_prices_are_carried_through(self):
        self._write("1920", [{
            "Date": "11/08/2019", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "PC>2.5": 2.0, "PC<2.5": 2.0, "MaxC>2.5": 2.15, "MaxC<2.5": 2.05,
        }])
        out = self._load(["1920"]).iloc[0]
        self.assertAlmostEqual(out["pin_odds_over"], 2.0)
        self.assertAlmostEqual(out["best_odds_over"], 2.15)  # best price beats Pinnacle

    def test_season_without_totals_odds_is_skipped(self):
        self._write("1819", [{
            "Date": "11/08/2018", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
        }])
        self._write("1920", [{
            "Date": "11/08/2019", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
            "PC>2.5": 2.0, "PC<2.5": 2.0, "MaxC>2.5": 2.1, "MaxC<2.5": 2.05,
        }])
        out = self._load(["1819", "1920"])
        self.assertEqual(set(out["season"]), {"2019-20"})


if __name__ == "__main__":
    unittest.main()
