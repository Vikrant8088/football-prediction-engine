"""Tests for the foreign-league cold start.

The dangerous failure here is not a missing player — it is a WRONG one. Handing one
footballer's finishing to another produces a confident, well-formed projection that
the optimiser will happily buy. So the refusal paths are pinned as hard as the
matching ones.
"""

import unittest

import pandas as pd

from research.data import foreign_rates as fr


def _row(name, minutes=1800, xg=18.0, xa=9.0, team="Valencia", yellows=0, reds=0,
         player_id=None):
    # Understat's player id is the identity: it separates one player who moved
    # mid-season from two different players who happen to share a name.
    return {"id": player_id or ("us-%s" % name.lower().replace(" ", "")),
            "player_name": name, "time": minutes, "xG": xg, "xA": xa,
            "team_title": team, "yellow_cards": yellows, "red_cards": reds}


class TestPreviousStartYear(unittest.TestCase):
    def test_july_points_at_the_season_just_finished(self):
        from datetime import datetime, timezone
        # July 2026: the season just finished is 2025/26, which Understat calls 2025.
        self.assertEqual(
            fr.previous_start_year(datetime(2026, 7, 17, tzinfo=timezone.utc)), 2025)

    def test_midseason_points_at_the_season_before(self):
        from datetime import datetime, timezone
        self.assertEqual(
            fr.previous_start_year(datetime(2027, 2, 1, tzinfo=timezone.utc)), 2025)


class TestForeignRates(unittest.TestCase):
    def setUp(self):
        self._real = fr.load_league_players

    def tearDown(self):
        fr.load_league_players = self._real

    def _patch(self, by_league):
        fr.load_league_players = lambda league, year: by_league.get(league, [])

    def test_rates_are_scaled_by_the_measured_transfer_ratio(self):
        # 18 xG in 1800 min = 0.9 per 90, expressed in PL terms by XG_TRANSFER_RATIO.
        self._patch({"La_liga": [_row("Test Striker", minutes=1800, xg=18.0, xa=9.0)]})
        rates = fr.foreign_rates(2025, leagues=("La_liga",))
        row = rates[fr.squash("Test Striker")]
        self.assertAlmostEqual(row["xg_per_90"], 0.9 * fr.XG_TRANSFER_RATIO, places=6)
        self.assertAlmostEqual(row["xa_per_90"], 0.45 * fr.XA_TRANSFER_RATIO, places=6)

    def test_small_samples_are_excluded(self):
        # 200 minutes and one chance reads as a superstar; it is noise.
        self._patch({"La_liga": [_row("Cameo Man", minutes=200, xg=1.0)]})
        self.assertEqual(fr.foreign_rates(2025, leagues=("La_liga",)), {})

    def test_unmeasured_channels_are_zero_not_invented(self):
        # Understat measures no saves, bonus or defensive contributions.
        self._patch({"La_liga": [_row("Test Striker")]})
        row = fr.foreign_rates(2025, leagues=("La_liga",))[fr.squash("Test Striker")]
        for field in ("saves_per_90", "bonus_per_90", "dc_per_90"):
            self.assertEqual(row[field], 0.0)

    def test_a_midseason_move_sums_both_spells(self):
        # SAME understat id in two leagues: one player who moved. Judge him on the
        # whole season (3600 min, 18 xG), not on half of it, and label him by his
        # main spell.
        self._patch({
            "La_liga": [_row("Split Season", minutes=900, xg=9.0, team="Valencia",
                             player_id="us-1")],
            "Serie_A": [_row("Split Season", minutes=2700, xg=9.0, team="Napoli",
                             player_id="us-1")],
        })
        row = fr.foreign_rates(2025, leagues=("La_liga", "Serie_A"))[fr.squash("Split Season")]
        self.assertEqual(row["minutes"], 3600.0)
        self.assertEqual(row["team"], "Napoli")
        self.assertAlmostEqual(row["xg_per_90"], (18.0 / 40.0) * fr.XG_TRANSFER_RATIO, places=6)

    def test_two_players_sharing_a_name_are_dropped_entirely(self):
        # DIFFERENT understat ids: two footballers, one name. Keeping the one with
        # more minutes would silently hand his finishing to the other.
        self._patch({"La_liga": [
            _row("Same Name", minutes=1000, xg=2.0, team="Sevilla", player_id="us-1"),
            _row("Same Name", minutes=3000, xg=30.0, team="Roma", player_id="us-2"),
        ]})
        self.assertEqual(fr.foreign_rates(2025, leagues=("La_liga",)), {})


class TestMatchPlayers(unittest.TestCase):
    def setUp(self):
        self._real = fr.load_league_players
        fr.load_league_players = lambda league, year: (
            [_row("Viktor Gyokeres", player_id="us-gyo"),
             _row("Ambiguous Name", team="Sevilla", player_id="us-a1"),
             _row("Ambiguous Name", team="Roma", minutes=1700, player_id="us-a2")]
            if league == "La_liga" else [])

    def tearDown(self):
        fr.load_league_players = self._real

    def _players(self):
        return pd.DataFrame([
            {"id": 1, "web_name": "Gyökeres", "full_name": "Viktor Gyökeres", "minutes": 0},
            {"id": 2, "web_name": "Salah", "full_name": "Mohamed Salah", "minutes": 0},
            {"id": 3, "web_name": "Haaland", "full_name": "Erling Haaland", "minutes": 1200},
        ])

    def test_matches_a_signing_with_no_pl_history(self):
        matched = fr.match_players(self._players(), fr.foreign_rates(2025, leagues=("La_liga",)))
        self.assertIn(1, matched)
        self.assertGreater(matched[1]["xg_per_90"], 0)

    def test_premier_league_history_always_wins(self):
        # Player 1 has real PL rates, so the foreign proxy must not override them.
        matched = fr.match_players(self._players(),
                                   fr.foreign_rates(2025, leagues=("La_liga",)),
                                   only_missing_from={1: {"xg_per_90": 0.5}})
        self.assertNotIn(1, matched)

    def test_a_player_already_playing_this_season_is_skipped(self):
        matched = fr.match_players(self._players(), fr.foreign_rates(2025, leagues=("La_liga",)))
        self.assertNotIn(3, matched)     # 1200 minutes already this season

    def test_unknown_player_is_simply_absent(self):
        matched = fr.match_players(self._players(), fr.foreign_rates(2025, leagues=("La_liga",)))
        self.assertNotIn(2, matched)     # Salah is not in the foreign feed

    def test_ambiguous_name_is_refused_not_guessed(self):
        players = pd.DataFrame([{"id": 9, "web_name": "Ambiguous Name",
                                 "full_name": "Ambiguous Name", "minutes": 0}])
        matched = fr.match_players(players, fr.foreign_rates(2025, leagues=("La_liga",)))
        self.assertEqual(matched, {}, "two players share this name — refuse, never guess")


if __name__ == "__main__":
    unittest.main()
