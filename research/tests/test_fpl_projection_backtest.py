"""Unit + leakage tests for the multi-season FPL projection backtest.

The headline claim of Phase 5c rests entirely on this backtest being honest, so
the tests below attack the two ways it could lie:

  1. TEMPORAL LEAKAGE - the team model seeing a match that had not been played,
     or a player's rates including the gameweek being predicted.
  2. DOUBLE COUNTING - a player with two fixtures in one gameweek being picked
     twice in a single XI, inflating our score.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import research.evaluation.benchmark_fpl_projections as backtest
from research.evaluation.benchmark_fpl_projections import (
    _mean_keepers_in_naive_xi,
    _rates_from_history,
    _top11,
    _verdict,
    compare_top11,
    run_season,
)

SEASON = "2022-23"
UNIFORM_GRID = np.full((6, 6), 1.0 / 36.0)


def _fixture(player_id, name, position, gameweek, team, opponent, kickoff,
             points, minutes=90, was_home=True, price=8.0, xg=0.3, xa=0.2):
    return {
        "season": SEASON, "gameweek": gameweek, "player_id": player_id, "player": name,
        "position": position, "team": team, "opponent": opponent,
        "was_home": was_home, "kickoff_time": pd.Timestamp(kickoff, tz="UTC"),
        "minutes": minutes, "expected_goals": xg, "expected_assists": xa,
        "saves": 0, "bonus": 0, "defensive_contribution": 0,
        "yellow_cards": 0, "red_cards": 0, "total_points": points, "price": price,
    }


def _synthetic_matches():
    """Four seasons of A-vs-B matches. The team model needs 3 before it will fit."""
    rows = []
    for season, year in [("2019-20", 2019), ("2020-21", 2020), ("2021-22", 2021)]:
        for month in (9, 10, 11):
            rows.append({"date": pd.Timestamp(f"{year}-{month:02d}-10"), "season": season,
                         "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1,
                         "home_xg": 1.8, "away_xg": 1.1})
            rows.append({"date": pd.Timestamp(f"{year}-{month:02d}-20"), "season": season,
                         "home_team": "B", "away_team": "A", "home_goals": 1, "away_goals": 1,
                         "home_xg": 1.2, "away_xg": 1.3})
    # Current season: one match BEFORE the GW6 cutoff, one AFTER it.
    rows.append({"date": pd.Timestamp("2022-08-20"), "season": SEASON,
                 "home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 0,
                 "home_xg": 2.4, "away_xg": 0.4})
    rows.append({"date": pd.Timestamp("2022-12-01"), "season": SEASON,
                 "home_team": "B", "away_team": "A", "home_goals": 5, "away_goals": 0,
                 "home_xg": 4.0, "away_xg": 0.2})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _synthetic_gameweeks():
    """One player per team, gameweeks 1..6. Gameweek 6 is a DOUBLE for player 1."""
    rows = []
    for gameweek in range(1, 6):
        kickoff = f"2022-09-{gameweek:02d}T12:00:00"
        rows.append(_fixture(1, "Striker", 4, gameweek, "A", "B", kickoff, points=5))
        rows.append(_fixture(2, "Keeper", 1, gameweek, "B", "A", kickoff, points=2,
                             was_home=False))
    # GW6: player 1 plays twice; the second fixture kicks off later.
    rows.append(_fixture(1, "Striker", 4, 6, "A", "B", "2022-09-10T12:00:00", points=7))
    rows.append(_fixture(1, "Striker", 4, 6, "A", "B", "2022-09-13T12:00:00", points=4,
                         was_home=False))
    rows.append(_fixture(2, "Keeper", 1, 6, "B", "A", "2022-09-10T12:00:00", points=3,
                         was_home=False))
    return pd.DataFrame(rows)


class _SpyEnsemble:
    """Records exactly which matches the team model was allowed to see."""

    seen = []

    def fit(self, train):
        _SpyEnsemble.seen.append(train.copy())
        return self

    def scoreline_grid(self, home, away):
        return UNIFORM_GRID


class TestRatesFromHistory(unittest.TestCase):
    def test_per_90_rates_use_only_supplied_rows(self):
        rows = [
            {"minutes": 90, "expected_goals": 0.5, "expected_assists": 0.25, "saves": 3,
             "bonus": 1, "defensive_contribution": 12, "yellow_cards": 1, "red_cards": 0},
            {"minutes": 90, "expected_goals": 0.5, "expected_assists": 0.25, "saves": 3,
             "bonus": 1, "defensive_contribution": 0, "yellow_cards": 0, "red_cards": 1},
        ]
        rates = _rates_from_history(rows, gameweeks_elapsed=2)
        self.assertEqual(rates["minutes"], 180)
        self.assertAlmostEqual(rates["xg_per_90"], 0.5)
        self.assertAlmostEqual(rates["xa_per_90"], 0.25)
        self.assertAlmostEqual(rates["saves_per_90"], 3.0)
        self.assertAlmostEqual(rates["dc_per_90"], 6.0)
        self.assertAlmostEqual(rates["cards_per_90"], 2.0)   # 1 yellow + 3*1 red, over 2x90

    def test_zero_minutes_never_divides_by_zero(self):
        rows = [{"minutes": 0, "expected_goals": 0.0, "expected_assists": 0.0, "saves": 0,
                 "bonus": 0, "defensive_contribution": 0, "yellow_cards": 0, "red_cards": 0}]
        rates = _rates_from_history(rows, gameweeks_elapsed=1)
        self.assertEqual(rates["xg_per_90"], 0.0)

    def test_gameweeks_elapsed_never_zero(self):
        self.assertEqual(_rates_from_history([], gameweeks_elapsed=0)["gameweeks"], 1)


class TestRunSeasonIsPointInTime(unittest.TestCase):
    def setUp(self):
        _SpyEnsemble.seen = []
        patcher = patch.object(backtest, "ScorelineEnsemble", _SpyEnsemble)
        self.addCleanup(patcher.stop)
        patcher.start()

        frame = _synthetic_gameweeks()
        loader = patch.object(backtest, "load_gameweeks", return_value=frame)
        self.addCleanup(loader.stop)
        loader.start()

        self.matches = _synthetic_matches()
        self.predictions = run_season(SEASON, self.matches)

    def test_team_model_never_sees_a_match_at_or_after_kickoff(self):
        self.assertTrue(_SpyEnsemble.seen, "the model was never fitted")
        cutoff = pd.Timestamp("2022-09-10T12:00:00")   # GW6's FIRST kickoff
        for train in _SpyEnsemble.seen:
            self.assertTrue((train["date"] < cutoff).all())
            # The 2022-12-01 thrashing is in the future and must be invisible.
            self.assertNotIn(pd.Timestamp("2022-12-01"), set(train["date"]))
            self.assertIn(pd.Timestamp("2022-08-20"), set(train["date"]))

    def test_only_gameweek_6_is_scored(self):
        # FIRST_SCORED_GAMEWEEK=6, so GW1-5 build history and are never predicted.
        self.assertEqual({p["gameweek"] for p in self.predictions}, {6})

    def test_double_gameweek_is_one_row_with_summed_actual(self):
        striker = [p for p in self.predictions if p["player_id"] == 1]
        self.assertEqual(len(striker), 1)
        self.assertEqual(striker[0]["actual"], 11)   # 7 + 4, not two rows of 7 and 4

    def test_double_gameweek_projection_sums_both_fixtures(self):
        keeper = next(p for p in self.predictions if p["player_id"] == 2)
        striker = next(p for p in self.predictions if p["player_id"] == 1)
        # Same player, two fixtures -> strictly more expected points than one.
        self.assertGreater(striker["ours"], 0)
        self.assertGreater(keeper["ours"], 0)

    def test_baselines_exclude_the_gameweek_being_predicted(self):
        striker = next(p for p in self.predictions if p["player_id"] == 1)
        # GW1-5 were 5 points each; GW6's 11 must not leak into the average.
        self.assertAlmostEqual(striker["player_ppg"], 5.0)
        self.assertAlmostEqual(striker["player_form5"], 5.0)

    def test_price_is_taken_from_the_predicted_gameweek(self):
        self.assertTrue(all(p["price"] == 8.0 for p in self.predictions))


def _squad_frame(season="2022-23", gameweek=1):
    """24 players: 4 GKP, 8 DEF, 8 MID, 4 FWD across 4 clubs.

    `ours` ranks perfectly (actual == projected); `bad` ranks backwards;
    `keeper_lover` rates every goalkeeper above every outfielder, which the naive
    top-11 would happily field and a legal XI must refuse.
    """
    rows, player = [], 0
    for position, count in ((1, 4), (2, 8), (3, 8), (4, 4)):
        for i in range(count):
            rows.append({
                "season": season, "gameweek": gameweek, "player_id": player,
                "position": position, "club": "club_%d" % (player % 4),
                "actual": player, "ours": player, "bad": -player,
                "keeper_lover": 100 - player if position == 1 else -player,
            })
            player += 1
    return pd.DataFrame(rows)


class TestLegalXiSelection(unittest.TestCase):
    def test_xi_has_exactly_one_goalkeeper(self):
        xi = backtest.select_legal_xi(_squad_frame(), "keeper_lover")
        self.assertEqual(len(xi), 11)
        self.assertEqual((xi["position"] == 1).sum(), 1)

    def test_xi_obeys_formation_bounds(self):
        xi = backtest.select_legal_xi(_squad_frame(), "ours")
        counts = xi["position"].value_counts()
        self.assertEqual(counts.get(1, 0), 1)
        self.assertTrue(3 <= counts.get(2, 0) <= 5)
        self.assertTrue(2 <= counts.get(3, 0) <= 5)
        self.assertTrue(1 <= counts.get(4, 0) <= 3)

    def test_xi_never_takes_more_than_three_from_one_club(self):
        xi = backtest.select_legal_xi(_squad_frame(), "ours")
        self.assertLessEqual(xi["club"].value_counts().max(), backtest.MAX_PER_CLUB)

    def test_naive_metric_would_field_many_goalkeepers(self):
        # The defect that invalidated the single-season Phase 5b result.
        frame = _squad_frame()
        naive = frame.nlargest(11, "keeper_lover")
        self.assertGreater((naive["position"] == 1).sum(), 1)
        self.assertEqual(_mean_keepers_in_naive_xi(frame, "keeper_lover"), 4.0)
        self.assertEqual(_mean_keepers_in_naive_xi(frame, "ours"), 0.0)

    def test_no_legal_xi_when_a_position_is_missing(self):
        outfielders = _squad_frame()
        outfielders = outfielders[outfielders["position"] != 1]   # no goalkeeper
        self.assertIsNone(backtest.select_legal_xi(outfielders, "ours"))

    def test_all_enumerated_formations_are_legal(self):
        for defenders, midfielders, forwards in backtest.FORMATIONS:
            self.assertEqual(1 + defenders + midfielders + forwards, 11)
            self.assertTrue(3 <= defenders <= 5)
            self.assertTrue(2 <= midfielders <= 5)
            self.assertTrue(1 <= forwards <= 3)
        self.assertEqual(len(backtest.FORMATIONS), 8)


class TestTop11AndComparison(unittest.TestCase):
    def _frame(self):
        return pd.concat([_squad_frame(season, gameweek)
                          for season in ("2022-23", "2023-24")
                          for gameweek in (1, 2)], ignore_index=True)

    def test_top11_is_grouped_by_season_and_gameweek(self):
        top = _top11(self._frame(), "ours")
        self.assertEqual(len(top), 4)                      # 2 seasons x 2 gameweeks

    def test_top11_skips_gameweeks_with_too_few_players(self):
        thin = _squad_frame().head(5)
        self.assertEqual(len(_top11(thin, "ours")), 0)

    def test_perfect_ranking_beats_inverted_ranking(self):
        result = compare_top11(self._frame(), "ours", "bad")
        self.assertGreater(result["mean_gain_per_gw"], 0)
        self.assertEqual(result["gameweeks_won"], 4)
        self.assertEqual(result["gameweeks"], 4)

    def test_significance_requires_both_tests(self):
        base = {"mean_gain_per_gw": 1.0, "gameweeks_won": 3, "gameweeks": 4,
                "season_gain_per_38_gw": 38.0}
        only_one = dict(base, paired_t_p=0.04, wilcoxon_p=0.20, significant=False)
        seasons = {"2022-23": dict(only_one)}
        self.assertIn("not", _verdict(only_one, seasons).lower())

        both = dict(base, paired_t_p=0.01, wilcoxon_p=0.02, significant=True)
        self.assertIn("real, usable edge", _verdict(both, {"2022-23": dict(both)}))

    def test_verdict_reports_a_loss_honestly(self):
        losing = {"mean_gain_per_gw": -2.0, "gameweeks_won": 1, "gameweeks": 4,
                  "season_gain_per_38_gw": -76.0, "paired_t_p": 0.01,
                  "wilcoxon_p": 0.01, "significant": False}
        verdict = _verdict(losing, {"2022-23": dict(losing)})
        self.assertIn("naive baseline picks a better XI", verdict)
        self.assertIn("negative result", verdict)


if __name__ == "__main__":
    unittest.main()
