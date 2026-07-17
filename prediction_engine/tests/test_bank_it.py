"""Tests for the Bank-It gameweek assembler (docs/04_BANK_IT_PIPELINE.md).

The hard parts (projection, squad optimisation) are validated elsewhere. What is
new here is ASSEMBLY: turning per-fixture projections into the single
optimizer-ready gameweek frame, correctly for double and blank gameweeks. These
tests pin exactly that, with an injected fake projector so no engine is needed.
"""

import json
import tempfile
import unittest
from pathlib import Path

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
        def projector(engine, players, home, away, minutes_history=None, start_pct=None):
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
        # Exactly one vice-captain, in the XI, and not the captain.
        self.assertEqual(sum(1 for r in artifact["xi"] if r["vice_captain"]), 1)
        self.assertIn("vice_captain_id", artifact)
        self.assertNotEqual(artifact["captain_id"], artifact["vice_captain_id"])
        # Must round-trip through JSON (no numpy types leaking in).
        json.dumps(artifact)
        # And render without error.
        self.assertIn("Starting XI", bank_it.render_markdown(artifact))


class TestLineupSnapshotSelection(unittest.TestCase):
    """The pre-deadline filter is an integrity control, not a convenience.

    A snapshot fetched after the deadline knows things we could not have known when
    the squad was locked — late team news, even the confirmed XI. Reading one would
    silently turn the forward test into hindsight and void the claim Bank-It exists
    to make. So this is pinned hard.
    """

    DEADLINE = "2026-08-21T17:30:00Z"

    def _write(self, folder, fetched_at, teams=20):
        path = Path(folder) / ("GW01_%s.json" % fetched_at.replace(":", "").replace("-", ""))
        path.write_text(json.dumps({
            "source": "test", "fetched_at": fetched_at, "season": "2026-27",
            "gameweek": 1, "team_count": teams, "teams": [],
        }), encoding="utf-8")
        return path

    def test_picks_the_newest_snapshot_before_the_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = Path(tmp) / "2026-27"
            season_dir.mkdir()
            self._write(season_dir, "2026-08-19T09:00:00Z")
            self._write(season_dir, "2026-08-21T09:00:00Z")     # newest, still before
            found = bank_it.latest_snapshot_before(self.DEADLINE, "2026-27", Path(tmp))
            self.assertIsNotNone(found)
            self.assertEqual(found[0]["fetched_at"], "2026-08-21T09:00:00Z")

    def test_never_reads_a_snapshot_published_after_the_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = Path(tmp) / "2026-27"
            season_dir.mkdir()
            self._write(season_dir, "2026-08-20T09:00:00Z")     # legitimate
            self._write(season_dir, "2026-08-21T18:00:00Z")     # AFTER the deadline
            found = bank_it.latest_snapshot_before(self.DEADLINE, "2026-27", Path(tmp))
            self.assertEqual(found[0]["fetched_at"], "2026-08-20T09:00:00Z",
                             "a post-deadline snapshot is hindsight and must be ignored")

    def test_returns_none_when_every_snapshot_is_too_late(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = Path(tmp) / "2026-27"
            season_dir.mkdir()
            self._write(season_dir, "2026-08-21T18:00:00Z")
            self.assertIsNone(
                bank_it.latest_snapshot_before(self.DEADLINE, "2026-27", Path(tmp)))

    def test_missing_archive_degrades_rather_than_raises(self):
        # A missing feed must never block the squad.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                bank_it.latest_snapshot_before(self.DEADLINE, "2026-27", Path(tmp)))
            start_pct, provenance = bank_it.lineup_start_pct(
                pd.DataFrame([{"id": 1, "web_name": "X", "full_name": "X Y",
                               "team": "Arsenal"}]),
                "2026-27", self.DEADLINE, Path(tmp))
            self.assertEqual(start_pct, {})
            self.assertIsNone(provenance)


class TestFramePassesStartPct(unittest.TestCase):
    def test_start_pct_reaches_the_projector(self):
        seen = {}

        def projector(engine, players, home, away, minutes_history=None, start_pct=None):
            seen["start_pct"] = start_pct
            return pd.DataFrame([_fixture_row(1, "A", "T1", 3, 5.0, 4.0, "T2")])

        bank_it.build_gameweek_frame(engine=None, players=None,
                                     fixtures=[("T1", "T2")],
                                     start_pct={1: 90}, projector=projector)
        self.assertEqual(seen["start_pct"], {1: 90})


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

    def test_opening_weeks_use_last_season_not_a_two_game_average(self):
        # docs/05 pre-registers this: through GW5 the baseline is last season's ppg,
        # even for a player who already has a (noisy) current-season average.
        ppg = bank_it.baseline_ppg(self._players(), minutes_history={1: [90, 90]},
                                   prior_ppg={1: 6.0}, gameweek=3)
        self.assertAlmostEqual(ppg[1], 6.0, msg="GW1-5 must use last season's ppg")

    def test_switches_to_the_current_season_after_the_opening_weeks(self):
        ppg = bank_it.baseline_ppg(self._players(), minutes_history={1: [90] * 20},
                                   prior_ppg={1: 6.0},
                                   gameweek=bank_it.EARLY_SEASON_GAMEWEEKS + 1)
        self.assertAlmostEqual(ppg[1], 5.0, msg="from GW6 the current season takes over")

    def test_gameweek_one_baseline_is_not_degenerate(self):
        # THE case this exists for. At GW1 nobody has a current-season average, so
        # without last-season ppg every player ties on zero and the baseline squad is
        # meaningless — which would void the pre-registered comparison at GW1.
        prior = {1: 5.0, 2: 4.0, 3: 3.5}
        ppg = bank_it.baseline_ppg(self._players(), minutes_history={},
                                   prior_ppg=prior, gameweek=1)
        self.assertEqual(ppg, {1: 5.0, 2: 4.0, 3: 3.5})
        self.assertGreater(len(set(ppg.values())), 1, "the baseline must discriminate")


class TestPreviousSeason(unittest.TestCase):
    def test_steps_back_one_season(self):
        self.assertEqual(bank_it.previous_season("2026-27"), "2025-26")
        self.assertEqual(bank_it.previous_season("2020-21"), "2019-20")


class TestLastSeasonPpgJoin(unittest.TestCase):
    def test_joins_on_the_stable_code_not_the_reassigned_id(self):
        # FPL reassigns element ids every summer. Here last season's id 7 and this
        # season's id 99 are the same human (code 154561); id 7 this season is someone
        # else entirely. An id join would hand that player the wrong history.
        players = pd.DataFrame([{"id": 99, "code": 154561},
                                {"id": 7, "code": 999999}])
        gw = pd.DataFrame([
            {"player_id": 7, "gameweek": 1, "total_points": 10},
            {"player_id": 7, "gameweek": 2, "total_points": 6},
        ])
        meta = {7: {"code": 154561}}

        import research.data.fpl_archive as archive
        real_gw, real_meta = archive.load_gameweeks, archive.load_player_meta
        archive.load_gameweeks = lambda season, *a, **k: gw
        archive.load_player_meta = lambda season, *a, **k: meta
        try:
            prior = bank_it.last_season_ppg(players, season="2025-26")
        finally:
            archive.load_gameweeks, archive.load_player_meta = real_gw, real_meta

        self.assertEqual(prior, {99: 8.0})       # (10 + 6) / 2 gameweeks, to id 99
        self.assertNotIn(7, prior)               # NOT to whoever now wears id 7


if __name__ == "__main__":
    unittest.main()
