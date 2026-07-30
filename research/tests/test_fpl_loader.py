"""Contract tests for the live FPL feed.

`fpl_loader` is the boundary between our engine and somebody else's API — the one
input we do not control and cannot version. FPL rewrites its bootstrap between
seasons; a renamed or dropped field would surface as a KeyError, and the moment it
would surface is the moment we run the squad before a deadline.

These assert the contract in two layers:

  parsing   against a fixture, so the transformations (price/10, canonical team names,
            availability, per-90 rates, the permanent `code`) are pinned without a
            network call and run anywhere, including CI.
  live      against whatever is actually ingested, skipped when absent. This is what
            fails loudly and early if FPL changes the shape under us.

Every field asserted here is one the projection genuinely consumes. A test that
merely restated the JSON would pass forever and protect nothing.
"""

import json
import unittest

from research.data import fpl_loader

# The fields the engine actually reads. Derived from the loader itself; if a field is
# added there without being considered here, that is worth noticing.
REQUIRED_ELEMENT_FIELDS = (
    "id", "code", "web_name", "first_name", "second_name", "team", "element_type",
    "now_cost", "minutes", "starts", "total_points", "goals_scored", "assists",
    "saves", "bonus", "yellow_cards", "red_cards", "expected_goals",
    "expected_assists", "status", "chance_of_playing_next_round",
)


def _element(**overrides):
    base = {
        "id": 1, "code": 154561, "web_name": "Raya", "first_name": "David",
        "second_name": "Raya Martin", "team": 1, "element_type": 1,
        "now_cost": 55, "minutes": 900, "starts": 10, "total_points": 60,
        "goals_scored": 0, "assists": 1, "saves": 45, "bonus": 5,
        "yellow_cards": 2, "red_cards": 0, "defensive_contribution": 30,
        "expected_goals": "0.10", "expected_assists": "0.50",
        "status": "a", "chance_of_playing_next_round": None,
    }
    base.update(overrides)
    return base


def _payload(elements=None, teams=None, events=None):
    return {
        "elements": elements if elements is not None else [_element()],
        "teams": teams if teams is not None else [
            {"id": 1, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "name": "Spurs", "short_name": "TOT"},
        ],
        "events": events if events is not None else [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z",
             "is_next": True, "finished": False},
        ],
    }


class TestParsingContract(unittest.TestCase):
    """Pinned against a fixture — no network, no ingested lake."""

    def setUp(self):
        self._real = fpl_loader._load_dataset

    def tearDown(self):
        fpl_loader._load_dataset = self._real

    def _patch(self, bootstrap=None, fixtures=None):
        def loader(dataset):
            if dataset == "bootstrap-static":
                return bootstrap if bootstrap is not None else _payload()
            if dataset == "fixtures":
                return fixtures if fixtures is not None else []
            raise ValueError(dataset)
        fpl_loader._load_dataset = loader

    def test_price_is_converted_from_tenths(self):
        self._patch()
        self.assertAlmostEqual(fpl_loader.load_players()["price"].iloc[0], 5.5)

    def test_permanent_code_is_carried_through(self):
        # The only safe key for joining across seasons; ids are reassigned each summer.
        self._patch()
        self.assertEqual(int(fpl_loader.load_players()["code"].iloc[0]), 154561)

    def test_per_90_rates_are_derived_not_copied(self):
        self._patch()
        row = fpl_loader.load_players().iloc[0]
        # 0.10 expected goals over 900 minutes = 10 per-90 units -> 0.01 per 90.
        self.assertAlmostEqual(row["xg_per_90"], 0.10 / (900 / 90.0))

    def test_a_player_with_no_minutes_gets_zero_rates_not_a_crash(self):
        self._patch(_payload([_element(minutes=0)]))
        self.assertEqual(fpl_loader.load_players()["xg_per_90"].iloc[0], 0.0)

    def test_availability_reads_status_not_chance(self):
        self._patch(_payload([_element(id=1, status="a"),
                              _element(id=2, status="i", chance_of_playing_next_round=0)]))
        players = fpl_loader.load_players().set_index("id")
        self.assertTrue(bool(players.loc[1, "available"]))
        self.assertFalse(bool(players.loc[2, "available"]))
        self.assertEqual(players.loc[1, "chance_of_playing"], 100.0)  # null -> fully fit
        self.assertEqual(players.loc[2, "chance_of_playing"], 0.0)

    def test_team_names_are_canonicalised_for_the_engine(self):
        # FPL says "Spurs"; the engine is trained on Understat's "Tottenham".
        self._patch()
        self.assertIn("Tottenham", set(fpl_loader.load_teams()["team"]))

    def test_ipswich_town_maps_to_ipswich(self):
        # Understat drops the "Town", and Ipswich have a real 2024/25 Premier League
        # season on record. Without this mapping the engine cold-starts them to league
        # average and throws away a season of results (found in the 2026/27 refresh).
        self.assertEqual(fpl_loader.canonical_team("Ipswich Town"), "Ipswich")

    def test_genuinely_new_clubs_pass_through_unmapped(self):
        # Coventry/Hull have no usable Understat history, so they correctly stay
        # unmapped and cold-start at the engine rather than being forced onto a proxy.
        self.assertEqual(fpl_loader.canonical_team("Coventry City"), "Coventry City")
        self.assertEqual(fpl_loader.canonical_team("Hull City"), "Hull City")

    def test_next_gameweek_prefers_the_is_next_flag(self):
        self._patch(_payload(events=[
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "is_next": False,
             "finished": True},
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "is_next": True,
             "finished": False},
        ]))
        self.assertEqual(fpl_loader.next_gameweek()["gameweek"], 2)

    def test_next_gameweek_is_none_between_seasons(self):
        # Every gameweek finished and none flagged next: the new game is not published.
        self._patch(_payload(events=[
            {"id": 1, "deadline_time": "2025-08-15T17:30:00Z", "is_next": False,
             "finished": True},
        ]))
        self.assertIsNone(fpl_loader.next_gameweek())

    def test_fixtures_resolve_team_ids_to_canonical_names(self):
        self._patch(fixtures=[{"id": 7, "event": 1, "team_h": 1, "team_a": 2,
                               "kickoff_time": "2026-08-21T19:00:00Z", "finished": False}])
        row = fpl_loader.load_fixtures().iloc[0]
        self.assertEqual((row["home_team"], row["away_team"]), ("Arsenal", "Tottenham"))
        self.assertEqual(fpl_loader.fixtures_for_gameweek(1), [("Arsenal", "Tottenham")])

    def test_unscheduled_fixture_keeps_a_null_gameweek(self):
        # FPL leaves `event` null until a fixture is scheduled; it must not become 0.
        self._patch(fixtures=[{"id": 8, "event": None, "team_h": 1, "team_a": 2,
                               "kickoff_time": None, "finished": False}])
        self.assertTrue(fpl_loader.load_fixtures()["gameweek"].isna().all())


class TestLiveFeedContract(unittest.TestCase):
    """Against the actually-ingested payload. Skips when nothing is ingested, so a
    clean checkout still passes — but fails loudly if FPL changes shape under us."""

    def setUp(self):
        try:
            self.payload = fpl_loader._load_dataset("bootstrap-static")
        except Exception as exc:
            self.skipTest("no ingested FPL bootstrap (%s)" % exc)

    def test_every_field_the_engine_reads_still_exists(self):
        element = self.payload["elements"][0]
        missing = [f for f in REQUIRED_ELEMENT_FIELDS if f not in element]
        self.assertEqual(missing, [],
                         "FPL dropped or renamed fields the projection depends on")

    def test_teams_all_map_to_names_the_engine_knows(self):
        from research.data.xg_loader import load_understat_matches
        try:
            matches = load_understat_matches("EPL")
        except Exception as exc:
            self.skipTest("no Understat matches ingested (%s)" % exc)
        known = set(matches["home_team"]) | set(matches["away_team"])
        unmapped = fpl_loader.unmapped_teams(known)
        # Promoted clubs legitimately have no Understat history yet, so this reports
        # rather than fails — but it must never be a silent surprise.
        if unmapped:
            print("\n  NOTE: %d FPL teams unknown to the engine: %s"
                  % (len(unmapped), ", ".join(unmapped)))

    def test_the_squad_is_a_plausible_size(self):
        players = fpl_loader.load_players()
        self.assertGreater(len(players), 400, "a Premier League season has ~600 players")
        self.assertEqual(players["position"].isin([1, 2, 3, 4]).all(), True)
        self.assertTrue((players["price"] > 3.0).all(), "no player costs under £3.5m")


if __name__ == "__main__":
    unittest.main()
