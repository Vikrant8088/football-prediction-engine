"""End-to-end integration test of the whole Bank-It chain.

Every other test here checks one component. This checks that they still fit together:

    projection -> gameweek frame -> squad + baseline + variant -> artifact
              -> JSON round-trip -> scoring -> ledger -> paired summary

That seam is where the real bugs have actually been — a projector gaining a keyword
argument, `start_pct` not threaded through, a variant scored on the wrong gameweeks.
Each of those passed every unit test and would have been caught here.

Runs entirely on fixtures: a stub engine and a synthetic 20-player league, so it needs
no data lake, no network and no model fit, and is therefore safe to run in CI on every
push. It asserts the *wiring*, not the football.
"""

import json
import unittest

import numpy as np
import pandas as pd

from prediction_engine.fpl import bank_it, scorer


class _StubEngine:
    """Enough of a PredictionEngine for `project_fixture`: a scoreline grid and the
    match history `team_scoring_rates` reads."""

    def __init__(self, teams):
        rows = []
        for i, home in enumerate(teams):
            for away in teams:
                if home == away:
                    continue
                rows.append({"season": "2025-26", "home_team": home, "away_team": away,
                             "home_goals": 2 if i % 2 else 1, "away_goals": 1})
        self.matches = pd.DataFrame(rows)

    def scoreline_grid(self, home, away, allow_unseen=False):
        # A plausible 6x6 joint scoreline distribution, normalised.
        grid = np.array([[0.06, 0.05, 0.02, 0.01, 0.00, 0.00],
                         [0.09, 0.08, 0.03, 0.01, 0.00, 0.00],
                         [0.08, 0.07, 0.03, 0.01, 0.00, 0.00],
                         [0.05, 0.04, 0.02, 0.01, 0.00, 0.00],
                         [0.02, 0.02, 0.01, 0.00, 0.00, 0.00],
                         [0.01, 0.01, 0.00, 0.00, 0.00, 0.00]])
        return grid / grid.sum()


def _league(teams, per_team=8):
    """A synthetic FPL player table: enough bodies per position for a legal squad."""
    rows, pid = [], 0
    for team in teams:
        # 2 GKP, 2 DEF, 2 MID, 2 FWD per club -> a legal 2/5/5/3 squad is reachable
        # while respecting the 3-per-club cap.
        for position in (1, 1, 2, 2, 3, 3, 4, 4)[:per_team]:
            pid += 1
            rows.append({
                "id": pid, "code": 100000 + pid,
                "web_name": "P%d" % pid, "full_name": "Player %d" % pid,
                "team": team, "position": position,
                "price": 4.0 + (pid % 7) * 0.5,
                "minutes": 900, "starts": 10, "total_points": 40 + (pid % 30),
                "goals_scored": 2, "assists": 2, "saves": 10, "bonus": 3,
                "yellow_cards": 1, "red_cards": 0, "defensive_contribution": 20,
                "expected_goals": 3.0, "expected_assists": 2.0,
                "available": True, "chance_of_playing": 100.0,
                "xg_per_90": 0.10 + (pid % 9) * 0.05,
                "xa_per_90": 0.05 + (pid % 5) * 0.03,
                "saves_per_90": 1.0 if position == 1 else 0.0,
                "bonus_per_90": 0.1, "dc_per_90": 2.0 + (pid % 4),
                "cards_per_90": 0.1,
            })
    return pd.DataFrame(rows)


class TestBankItEndToEnd(unittest.TestCase):
    TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Brighton", "Fulham"]

    def setUp(self):
        self.engine = _StubEngine(self.TEAMS)
        self.players = _league(self.TEAMS)
        self.fixtures = [("Arsenal", "Chelsea"), ("Liverpool", "Everton"),
                         ("Brighton", "Fulham")]
        self.minutes_history = {int(p): [90, 90, 78, 90] for p in self.players["id"]}

    def _frame(self, start_pct=None):
        return bank_it.build_gameweek_frame(
            self.engine, self.players, self.fixtures,
            minutes_history=self.minutes_history, start_pct=start_pct)

    def test_the_whole_chain_runs_and_the_numbers_survive_it(self):
        frame = self._frame()
        self.assertEqual(len(frame), len(self.players),
                         "every player in a projected fixture should appear once")

        squad = bank_it.pick_squad(frame, budget=100.0)
        self.assertIsNotNone(squad, "a legal squad must exist for this league")

        # Baseline, exactly as bank_gameweek builds it.
        from prediction_engine.fpl.optimizer import select_squad
        frame["player_ppg"] = frame["player_id"].map(
            bank_it.baseline_ppg(self.players, self.minutes_history, gameweek=10))
        baseline = select_squad(frame, "player_ppg", squad_budget=100.0)
        self.assertIsNotNone(baseline, "the baseline squad must also be legal")

        artifact = bank_it.build_artifact(
            frame, squad, gameweek=10, deadline="2026-10-24T10:00:00Z",
            config={"budget": 100.0}, baseline_squad=baseline)

        # Must survive the round-trip that actually happens on disk.
        artifact = json.loads(json.dumps(artifact))
        self.assertEqual(len(artifact["xi"]), 11)
        self.assertEqual(len(artifact["bench"]), 4)
        self.assertEqual(sum(1 for r in artifact["xi"] if r["captain"]), 1)
        self.assertEqual(sum(1 for r in artifact["xi"] if r["vice_captain"]), 1)

        # Score it against invented actuals, then ledger it.
        actuals = {int(p): {"points": 2.0 + (int(p) % 9), "minutes": 90.0}
                   for p in self.players["id"]}
        record = scorer.score_artifact(artifact, actuals)
        self.assertIn("baseline", record)
        self.assertAlmostEqual(record["gain"], record["ours"] - record["baseline"])

        ledger = scorer.SeasonLedger("2026-27")
        ledger.add(record)
        summary = ledger.summary()
        self.assertEqual(summary["scored_gameweeks"], 1)
        self.assertEqual(summary["gameweeks"], 1)

    def test_squad_obeys_every_fpl_rule(self):
        """The optimiser is proven elsewhere; this checks we hand it a frame that
        makes it produce a squad a manager could actually register."""
        frame = self._frame()
        squad = bank_it.pick_squad(frame, budget=100.0)
        picked = frame.loc[list(squad.xi) + list(squad.bench)]

        self.assertEqual(len(picked), 15)
        counts = picked["position"].value_counts().to_dict()
        self.assertEqual({counts.get(1), counts.get(2), counts.get(3), counts.get(4)},
                         {2, 5, 5, 3}, "squad quota must be 2/5/5/3")
        self.assertLessEqual(picked["price"].sum(), 100.0 + 1e-9)
        self.assertLessEqual(picked["club"].value_counts().max(), 3,
                             "max 3 players per club, across the whole squad")

        xi = frame.loc[list(squad.xi)]
        self.assertEqual((xi["position"] == 1).sum(), 1, "exactly one goalkeeper starts")
        self.assertTrue(3 <= (xi["position"] == 2).sum() <= 5)
        self.assertTrue(1 <= (xi["position"] == 4).sum() <= 3)

    def test_the_lineup_feed_changes_the_squad_and_is_threaded_end_to_end(self):
        """The bug this exists for: `start_pct` silently not reaching the projector.
        Benching the entire highest-projected club must change who is picked."""
        base_frame = self._frame()
        base_squad = bank_it.pick_squad(base_frame, budget=100.0)
        base_xi = set(base_frame.loc[list(base_squad.xi), "player_id"])

        arsenal = self.players[self.players["team"] == "Arsenal"]["id"]
        start_pct = {int(p): 0 for p in arsenal}          # the feed says: none of them start

        fed_frame = self._frame(start_pct=start_pct)
        fed_squad = bank_it.pick_squad(fed_frame, budget=100.0)
        fed_xi = set(fed_frame.loc[list(fed_squad.xi), "player_id"])

        self.assertNotEqual(base_xi, fed_xi, "a zeroed club must change the XI")
        self.assertFalse(fed_xi & set(int(p) for p in arsenal),
                         "nobody the feed benched should start")

    def test_a_blank_gameweek_player_cannot_be_picked(self):
        # Fulham/Brighton removed from the fixture list -> their players are absent.
        frame = bank_it.build_gameweek_frame(
            self.engine, self.players, [("Arsenal", "Chelsea")],
            minutes_history=self.minutes_history)
        self.assertEqual(set(frame["team"]), {"Arsenal", "Chelsea"})

    def test_a_double_gameweek_sums_into_one_row(self):
        # Arsenal plays twice; each player must still appear exactly once.
        frame = bank_it.build_gameweek_frame(
            self.engine, self.players,
            [("Arsenal", "Chelsea"), ("Liverpool", "Arsenal")],
            minutes_history=self.minutes_history)
        arsenal = frame[frame["team"] == "Arsenal"]
        self.assertEqual(len(arsenal), len(arsenal["player_id"].unique()))
        # And a doubled player should out-project his single-fixture equivalent.
        single = frame[frame["team"] == "Chelsea"]["expected_points"].max()
        self.assertGreater(arsenal["expected_points"].max(), single)


if __name__ == "__main__":
    unittest.main()
