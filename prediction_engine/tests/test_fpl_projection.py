"""Tests for FPL fixture projections."""

import unittest

import numpy as np
import pandas as pd

from prediction_engine.fpl.projection import (
    _live_minutes_model,
    expected_minutes,
    fixture_context,
    project_player,
    team_scoring_rates,
)
from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID


def _grid(cells, size=4):
    g = np.zeros((size, size))
    for (h, a), p in cells.items():
        g[h, a] = p
    return g / g.sum()


def _player(**overrides):
    base = {
        "position": MID, "minutes": 38 * 90, "available": True, "chance_of_playing": 100.0,
        "xg_per_90": 0.0, "xa_per_90": 0.0, "saves_per_90": 0.0,
        "bonus_per_90": 0.0, "dc_per_90": 0.0, "cards_per_90": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


NEUTRAL_RATE = {"scored_per_match": 1.5, "conceded_per_match": 1.5}


class TestFixtureContext(unittest.TestCase):
    def setUp(self):
        # Home wins 2-0 half the time, 1-1 the other half.
        self.grid = _grid({(2, 0): 0.5, (1, 1): 0.5})

    def test_expected_goals_from_each_side(self):
        home = fixture_context(self.grid, is_home=True)
        away = fixture_context(self.grid, is_home=False)
        self.assertAlmostEqual(home["expected_goals_for"], 1.5)   # (2+1)/2
        self.assertAlmostEqual(home["expected_goals_against"], 0.5)
        self.assertAlmostEqual(away["expected_goals_for"], 0.5)
        self.assertAlmostEqual(away["expected_goals_against"], 1.5)

    def test_clean_sheet_probability(self):
        home = fixture_context(self.grid, is_home=True)
        self.assertAlmostEqual(home["clean_sheet_probability"], 0.5)  # away scores 0 half the time
        away = fixture_context(self.grid, is_home=False)
        self.assertAlmostEqual(away["clean_sheet_probability"], 0.0)  # home always scores

    def test_conceded_penalty_uses_floor_of_two(self):
        # Opponent scores 3 always -> floor(3/2) = 1 point deducted.
        grid = _grid({(0, 3): 1.0})
        ctx = fixture_context(grid, is_home=True)
        self.assertAlmostEqual(ctx["expected_conceded_penalty"], 1.0)


class TestExpectedMinutes(unittest.TestCase):
    def test_ever_present_plays_ninety(self):
        self.assertAlmostEqual(expected_minutes(_player(minutes=38 * 90)), 90.0)

    def test_availability_scales_minutes_down(self):
        p = _player(minutes=38 * 90, available=False, chance_of_playing=25.0)
        self.assertAlmostEqual(expected_minutes(p), 22.5)

    def test_injured_player_projects_zero(self):
        p = _player(minutes=38 * 90, available=False, chance_of_playing=0.0)
        self.assertAlmostEqual(expected_minutes(p), 0.0)


class TestProjectPlayer(unittest.TestCase):
    def setUp(self):
        self.clean_sheet_ctx = {
            "expected_goals_for": 1.5, "expected_goals_against": 0.0,
            "clean_sheet_probability": 1.0, "expected_conceded_penalty": 0.0,
        }

    def test_unavailable_player_scores_zero(self):
        p = _player(available=False, chance_of_playing=0.0, xg_per_90=1.0)
        self.assertEqual(project_player(p, self.clean_sheet_ctx, NEUTRAL_RATE)["expected_points"], 0.0)

    def test_ever_present_gets_two_appearance_points(self):
        out = project_player(_player(), self.clean_sheet_ctx, NEUTRAL_RATE)
        self.assertAlmostEqual(out["appearance"], 2.0)
        self.assertAlmostEqual(out["appearance_factor"], 1.0)

    def test_half_time_player_gets_about_one_appearance_point(self):
        p = _player(minutes=38 * 30)  # averages 30 minutes a match
        out = project_player(p, self.clean_sheet_ctx, NEUTRAL_RATE)
        self.assertAlmostEqual(out["appearance"], 1.0)

    def test_goal_points_scale_with_position(self):
        ctx = self.clean_sheet_ctx
        fwd = project_player(_player(position=FWD, xg_per_90=1.0), ctx, NEUTRAL_RATE)
        mid = project_player(_player(position=MID, xg_per_90=1.0), ctx, NEUTRAL_RATE)
        # Same xG, but a midfielder's goal is worth 5 and a forward's 4.
        self.assertGreater(mid["goals"], fwd["goals"])

    def test_defender_benefits_from_clean_sheet_probability(self):
        good = project_player(_player(position=DEF), self.clean_sheet_ctx, NEUTRAL_RATE)
        leaky_ctx = dict(self.clean_sheet_ctx, clean_sheet_probability=0.0)
        bad = project_player(_player(position=DEF), leaky_ctx, NEUTRAL_RATE)
        self.assertAlmostEqual(good["clean_sheet"], 4.0)
        self.assertAlmostEqual(bad["clean_sheet"], 0.0)
        self.assertGreater(good["expected_points"], bad["expected_points"])

    def test_easy_fixture_lifts_a_strikers_goals(self):
        easy = dict(self.clean_sheet_ctx, expected_goals_for=3.0)  # 2x the team's 1.5 average
        hard = dict(self.clean_sheet_ctx, expected_goals_for=0.75)
        p = _player(position=FWD, xg_per_90=0.5)
        self.assertAlmostEqual(
            project_player(p, easy, NEUTRAL_RATE)["expected_goals"], 1.0
        )
        self.assertAlmostEqual(
            project_player(p, hard, NEUTRAL_RATE)["expected_goals"], 0.25
        )

    def test_goalkeeper_saves_convert_at_three_per_point(self):
        gk = _player(position=GKP, saves_per_90=6.0)
        ctx = dict(self.clean_sheet_ctx, expected_goals_against=1.5)  # equals team average
        out = project_player(gk, ctx, NEUTRAL_RATE)
        self.assertAlmostEqual(out["saves"], 2.0)  # 6 saves / 3

    def test_defensive_points_use_a_threshold_not_a_rate(self):
        # A defender averaging 3 actions per 90 almost never reaches 10.
        low = project_player(_player(position=DEF, dc_per_90=3.0), self.clean_sheet_ctx, NEUTRAL_RATE)
        # One averaging 14 usually does.
        high = project_player(_player(position=DEF, dc_per_90=14.0), self.clean_sheet_ctx, NEUTRAL_RATE)
        self.assertLess(low["defensive"], 0.2)
        self.assertGreater(high["defensive"], 1.5)
        self.assertLessEqual(high["defensive"], 2.0)  # capped at the 2-point award


class TestTeamScoringRates(unittest.TestCase):
    def test_rates_are_per_match_over_the_latest_season(self):
        matches = pd.DataFrame([
            {"season": "2024-25", "home_team": "A", "away_team": "B", "home_goals": 3, "away_goals": 1},
            {"season": "2024-25", "home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 1},
            {"season": "2023-24", "home_team": "A", "away_team": "B", "home_goals": 9, "away_goals": 9},
        ])
        rates = team_scoring_rates(matches)
        # Only 2024-25 counts: A scored 3 + 1 = 4 over 2 matches.
        self.assertAlmostEqual(rates["A"]["scored_per_match"], 2.0)
        self.assertAlmostEqual(rates["A"]["conceded_per_match"], 0.5)


class TestLiveMinutesModel(unittest.TestCase):
    """The live wiring: recent-form when per-match minutes are supplied, with the
    availability flag folded in; crude fallback otherwise."""

    def _player(self, player_id=1, chance=100.0, available=True):
        return _player(id=player_id, chance_of_playing=chance, available=available)

    def test_no_history_falls_back_to_crude(self):
        self.assertIsNone(_live_minutes_model(self._player(), None))

    def test_player_absent_from_history_falls_back(self):
        self.assertIsNone(_live_minutes_model(self._player(player_id=7), {3: [90, 90]}))

    def test_recent_form_used_when_history_present(self):
        # A permanent substitute (always ~20 min): recent-form must zero his
        # clean-sheet eligibility, which the crude average would wrongly allow.
        m = _live_minutes_model(self._player(), {1: [20, 20, 20, 20, 20]})
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m["p_60"], 0.0)
        self.assertGreater(m["p_play"], 0.0)

    def test_injury_flag_zeroes_a_nailed_starter(self):
        # The live-only signal: a 0%-chance player is projected out even with a
        # perfect recent starting record.
        m = _live_minutes_model(self._player(chance=0.0, available=False),
                                {1: [90, 90, 90, 90]})
        self.assertAlmostEqual(m["expected_minutes"], 0.0)

    def test_doubtful_flag_scales_down(self):
        full = _live_minutes_model(self._player(chance=100.0), {1: [90, 90, 90]})
        doubt = _live_minutes_model(self._player(chance=50.0), {1: [90, 90, 90]})
        self.assertLess(doubt["expected_minutes"], full["expected_minutes"])


if __name__ == "__main__":
    unittest.main()
