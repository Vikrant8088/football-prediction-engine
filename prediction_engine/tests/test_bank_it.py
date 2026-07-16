"""Tests for the Bank-It gameweek assembler (docs/04_BANK_IT_PIPELINE.md).

The hard parts (projection, squad optimisation) are validated elsewhere. What is
new here is ASSEMBLY: turning per-fixture projections into the single
optimizer-ready gameweek frame, correctly for double and blank gameweeks. These
tests pin exactly that, with an injected fake projector so no engine is needed.
"""

import unittest

import pandas as pd

from prediction_engine.fpl import bank_it
from prediction_engine.fpl.optimizer import SquadSelection


def _fixture_row(player_id, player, team, position_id, price, expected_points,
                 opponent, available=True):
    """A single row shaped like one `project_fixture` output row."""
    return {
        "player_id": player_id,
        "player": player,
        "team": team,
        "position": {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[position_id],
        "position_id": position_id,
        "price": price,
        "opponent": opponent,
        "available": available,
        "appearance_factor": 1.0,
        "clean_sheet_probability": 0.3,
        "expected_points": expected_points,
        "goals": 0.0,
    }


class TestBuildGameweekFrame(unittest.TestCase):
    def _projector_from(self, fixture_rows):
        def projector(engine, players, home, away, minutes_history=None):
            return pd.DataFrame(fixture_rows[(home, away)])
        return projector

    def test_double_gameweek_sums_a_players_points_into_one_row(self):
        # Player 1 (team T1) plays twice this gameweek: T1 v T2 and T1 v T3.
        fixture_rows = {
            ("T1", "T2"): [
                _fixture_row(1, "Star", "T1", 3, 8.0, 5.0, "T2"),
                _fixture_row(2, "Foe", "T2", 3, 6.0, 3.0, "T1"),
            ],
            ("T1", "T3"): [
                _fixture_row(1, "Star", "T1", 3, 8.0, 4.0, "T3"),
                _fixture_row(3, "Other", "T3", 3, 6.0, 2.0, "T1"),
            ],
        }
        frame = bank_it.build_gameweek_frame(
            engine=None, players=None,
            fixtures=[("T1", "T2"), ("T1", "T3")],
            projector=self._projector_from(fixture_rows))

        star = frame[frame["player_id"] == 1]
        self.assertEqual(len(star), 1, "double gameweek must collapse to one row")
        self.assertAlmostEqual(float(star["expected_points"].iloc[0]), 9.0,
                               msg="a double gameweek's points are summed")

    def test_maps_position_to_int_and_adds_club_column(self):
        fixture_rows = {("T1", "T2"): [
            _fixture_row(1, "Keeper", "T1", 1, 5.0, 4.0, "T2"),
            _fixture_row(2, "Fwd", "T2", 4, 9.0, 6.0, "T1"),
        ]}
        frame = bank_it.build_gameweek_frame(
            engine=None, players=None, fixtures=[("T1", "T2")],
            projector=self._projector_from(fixture_rows))

        self.assertTrue(pd.api.types.is_integer_dtype(frame["position"]),
                        "optimizer needs an integer position column")
        self.assertIn("club", frame.columns)
        self.assertEqual(set(frame["club"]), {"T1", "T2"})
        # The human-readable name is preserved separately.
        self.assertIn("position_name", frame.columns)

    def test_blank_gameweek_player_is_absent(self):
        # T3's players are in no projected fixture, so none can be selected.
        fixture_rows = {("T1", "T2"): [
            _fixture_row(1, "A", "T1", 3, 5.0, 5.0, "T2"),
            _fixture_row(2, "B", "T2", 3, 5.0, 4.0, "T1"),
        ]}
        frame = bank_it.build_gameweek_frame(
            engine=None, players=None, fixtures=[("T1", "T2")],
            projector=self._projector_from(fixture_rows))
        self.assertEqual(set(frame["team"]), {"T1", "T2"})

    def test_empty_fixtures_raises(self):
        with self.assertRaises(ValueError):
            bank_it.build_gameweek_frame(engine=None, players=None, fixtures=[])


class TestPickSquadAndArtifact(unittest.TestCase):
    """A full optimizer-ready frame → legal squad → serialisable artifact."""

    def _full_frame(self):
        rows = []
        pid = 0
        # A 15-man squad with max 3 per club needs >=5 clubs; use 6 so no club is
        # ever over the cap (15 players / 6 clubs -> at most 3 each).
        plan = [(1, 2), (2, 5), (3, 5), (4, 3)]   # (position_id, count) = 2/5/5/3
        for position_id, count in plan:
            for i in range(count):
                pid += 1
                rows.append({
                    "player_id": pid,
                    "player": "P%d" % pid,
                    "team": "C%d" % (pid % 6),
                    "club": "C%d" % (pid % 6),
                    "position_name": {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}[position_id],
                    "position": position_id,
                    "position_id": position_id,
                    "price": 5.0,
                    "expected_points": float(pid),      # distinct, so ranking is total
                    "available": True,
                })
        return pd.DataFrame(rows)

    def test_pick_squad_returns_a_legal_squad(self):
        frame = self._full_frame()
        squad = bank_it.pick_squad(frame, budget=100.0)
        self.assertIsInstance(squad, SquadSelection)
        self.assertEqual(len(squad.xi), 11)
        self.assertEqual(len(squad.bench), 4)
        self.assertLessEqual(squad.cost, 100.0)
        self.assertIn(squad.captain, squad.xi)

    def test_artifact_is_serialisable_and_complete(self):
        import json
        frame = self._full_frame()
        squad = bank_it.pick_squad(frame, budget=100.0)
        artifact = bank_it.build_artifact(
            frame, squad, gameweek=1, deadline="2026-08-21T17:30:00Z",
            config={"budget": 100.0})

        self.assertEqual(len(artifact["xi"]), 11)
        self.assertEqual(len(artifact["bench"]), 4)
        self.assertEqual(sum(1 for r in artifact["xi"] if r["captain"]), 1)
        # Must round-trip through JSON (no numpy types leaking in).
        json.dumps(artifact)
        # And render without error.
        self.assertIn("Starting XI", bank_it.render_markdown(artifact))


class TestBaselinePpg(unittest.TestCase):
    def _players(self):
        return pd.DataFrame([
            {"id": 1, "total_points": 100.0},   # played 20 -> ppg 5.0
            {"id": 2, "total_points": 40.0},    # played 10 -> ppg 4.0
            {"id": 3, "total_points": 0.0},     # no history -> prior/0
        ])

    def test_ppg_is_points_over_gameweeks_played(self):
        ppg = bank_it.baseline_ppg(self._players(),
                                   minutes_history={1: [90] * 20, 2: [90] * 10})
        self.assertAlmostEqual(ppg[1], 5.0)
        self.assertAlmostEqual(ppg[2], 4.0)

    def test_no_history_uses_prior_then_zero(self):
        ppg = bank_it.baseline_ppg(self._players(), minutes_history={},
                                   prior_ppg={3: 3.5})
        self.assertAlmostEqual(ppg[3], 3.5)         # last-season ppg
        self.assertAlmostEqual(ppg[1], 0.0)         # no history, no prior -> 0


if __name__ == "__main__":
    unittest.main()
