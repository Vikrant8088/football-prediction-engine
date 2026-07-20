"""The live carried-squad tracks, and the controls that make them worth running.

A carried squad is STATEFUL, which the rebuild endpoint is not. That introduces
failure modes the rest of the pipeline cannot have, and each one would quietly
destroy the A/B rather than break it loudly:

  - re-deciding a gameweek after the deadline, with information that arrived after it
  - the two tracks opening from different squads, so they differ for reasons other
    than the transfer decision
  - state that round-trips through JSON incorrectly, silently changing the bank or
    the prices a player was bought at
  - hits scored gross, crediting a policy with points it paid to have

so each gets a test.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prediction_engine.fpl import carried
from prediction_engine.fpl.scorer import SeasonLedger, score_artifact
from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID

SEASON = "2026-27"


def _frame(bump=None):
    """A small legal universe: 24 players over 8 clubs, 2/5/5/3 reachable."""
    rows, pid = [], 0
    layout = [GKP, GKP, DEF, DEF, MID, MID, FWD, FWD]
    for club in range(8):
        for position in layout:
            pid += 1
            rows.append({
                "player_id": pid,
                "player": "P%d" % pid,
                "position_id": position,
                "club": "C%d" % club,
                "price": 4.5,
                "expected_points": round(1.0 + (pid % 11) * 0.37, 3),
                "player_ppg": round(1.0 + ((pid * 7) % 11) * 0.37, 3),
            })
    frame = pd.DataFrame(rows)
    if bump:
        for player_id, value in bump.items():
            frame.loc[frame["player_id"] == player_id, "expected_points"] = value
    return frame


def _opening_ids(frame):
    """A legal 2/5/5/3 fifteen, at most 3 per club."""
    chosen, per_club = [], {}
    for position, quota in ((GKP, 2), (DEF, 5), (MID, 5), (FWD, 3)):
        taken = 0
        for _, row in frame[frame["position_id"] == position].iterrows():
            if taken == quota:
                break
            club = row["club"]
            if per_club.get(club, 0) >= 3:
                continue
            per_club[club] = per_club.get(club, 0) + 1
            chosen.append(int(row["player_id"]))
            taken += 1
    return chosen


class _TempLive(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._real = carried.LIVE_DIR
        carried.LIVE_DIR = self.tmp

    def tearDown(self):
        carried.LIVE_DIR = self._real
        shutil.rmtree(str(self.tmp), ignore_errors=True)


class TestOpening(_TempLive):
    def test_both_tracks_open_from_the_identical_squad(self):
        """If they start apart, every later difference is confounded and the whole
        A/B measures nothing."""
        frame = _frame()
        blocks = carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        ours = carried.load_state(SEASON, "carried_ours")
        ppg = carried.load_state(SEASON, "carried_ppg")
        self.assertEqual(sorted(ours["players"]), sorted(ppg["players"]))
        self.assertEqual(ours["bank"], ppg["bank"])
        self.assertEqual(set(blocks), {"carried_ours", "carried_ppg"})

    def test_opening_makes_no_transfer(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        state = carried.load_state(SEASON, "carried_ours")
        self.assertEqual(state["history"][-1]["moves"], [])

    def test_the_opening_squad_is_bought_inside_the_budget(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        state = carried.load_state(SEASON, "carried_ours")
        spent = sum(state["bought"].values())
        self.assertEqual(spent + state["bank"], 1000, "£100.0m, in tenths")
        self.assertGreaterEqual(state["bank"], 0)


class TestRerunIsRefused(_TempLive):
    """The integrity control. A stateful track that can be re-decided after kickoff is
    worse than no track at all: it looks like a locked prediction and is not one."""

    def test_stepping_the_same_gameweek_twice_raises(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        state = carried.load_state(SEASON, "carried_ours")
        with self.assertRaises(ValueError):
            carried.step(frame, state, "carried_ours", SEASON, 1)

    def test_stepping_backwards_raises(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 5)
        state = carried.load_state(SEASON, "carried_ours")
        with self.assertRaises(ValueError):
            carried.step(frame, state, "carried_ours", SEASON, 4)

    def test_advancing_twice_reuses_the_locked_state_rather_than_re_deciding(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        first = carried.load_state(SEASON, "carried_ours")
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        self.assertEqual(carried.load_state(SEASON, "carried_ours"), first)


class TestAdvancing(_TempLive):
    def test_a_track_transfers_toward_its_own_value_column(self):
        """The point of the A/B: given a player who is a huge upgrade on OUR numbers
        and a nobody on ppg, only `carried_ours` should buy him."""
        frame = _frame()
        opening = _opening_ids(frame)
        carried.advance_tracks(frame, opening, SEASON, 1)

        # The target must be BUYABLE, not merely absent: a forward whose club has no
        # squad members at all, so the 3-per-club cap cannot block the transfer.
        # (First attempt at this test picked a forward whose club was already full —
        # the code correctly refused, and the fixture, not the code, was wrong.)
        squad_clubs = {}
        for pid in opening:
            club = frame.loc[frame["player_id"] == pid, "club"].iloc[0]
            squad_clubs[club] = squad_clubs.get(club, 0) + 1
        outsider = next(int(row["player_id"]) for _, row in frame.iterrows()
                        if int(row["player_id"]) not in opening
                        and row["position_id"] == FWD
                        and squad_clubs.get(row["club"], 0) == 0)
        later = _frame()
        later.loc[later["player_id"] == outsider, "expected_points"] = 99.0
        later.loc[later["player_id"] == outsider, "player_ppg"] = 0.0

        carried.advance_tracks(later, opening, SEASON, 2)
        ours = carried.load_state(SEASON, "carried_ours")
        ppg = carried.load_state(SEASON, "carried_ppg")
        self.assertIn(outsider, ours["players"], "our track should chase our own number")
        self.assertNotIn(outsider, ppg["players"], "ppg sees a nobody and must not buy")

    def test_state_round_trips_through_json_without_drift(self):
        frame = _frame()
        carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        path = carried.state_path(SEASON, "carried_ours")
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        squad = carried._squad_from_state(reloaded, SEASON)
        self.assertEqual(len(squad.players), 15)
        self.assertEqual(sum(squad.bought.values()) + squad.bank, 1000)

    def test_the_locked_block_is_a_legal_scoreable_team(self):
        frame = _frame()
        blocks = carried.advance_tracks(frame, _opening_ids(frame), SEASON, 1)
        block = blocks["carried_ours"]
        self.assertEqual(len(block["xi"]), 11)
        self.assertEqual(len(block["bench"]), 4)
        self.assertIn(block["captain_id"], block["xi"])
        self.assertIn(block["vice_captain_id"], block["xi"])
        self.assertNotEqual(block["captain_id"], block["vice_captain_id"])
        self.assertEqual(block["hits"], 0)


class TestScoringTracks(unittest.TestCase):
    def test_hits_are_subtracted_from_a_variant(self):
        """Scoring a carried track gross would credit it with the points it PAID to
        assemble the team, flattering exactly the policies that transfer most."""
        artifact = {
            "gameweek": 7, "captain_id": 1,
            "xi": [{"player_id": 1}, {"player_id": 2}],
            "variants": {
                "carried_ours": {"xi": [1, 2], "captain_id": 1, "hits": 4},
                "carried_ppg": {"xi": [1, 2], "captain_id": 1, "hits": 0},
            },
        }
        record = score_artifact(artifact, {1: 10.0, 2: 5.0})
        self.assertEqual(record["variants"]["carried_ours"]["gross_points"], 25.0)
        self.assertEqual(record["variants"]["carried_ours"]["points"], 21.0)
        self.assertEqual(record["variants"]["carried_ppg"]["points"], 25.0)

    def test_a_track_with_no_xi_is_skipped_not_scored_as_zero(self):
        # Scoring a failed track as 0 would look like a catastrophic gameweek and
        # would poison the paired test with a week that never happened.
        artifact = {"gameweek": 7, "captain_id": 1, "xi": [{"player_id": 1}],
                    "variants": {"carried_ours": {"name": "carried_ours",
                                                  "error": "no legal XI"}}}
        record = score_artifact(artifact, {1: 10.0})
        self.assertNotIn("carried_ours", record.get("variants", {}))

    def test_head_to_head_pairs_the_two_tracks_directly(self):
        ledger = SeasonLedger("2026-27")
        for gameweek, (a, b) in enumerate([(60, 50), (55, 52), (70, 61)], start=6):
            ledger.add({"gameweek": gameweek, "ours": 58, "baseline": 50,
                        "variants": {"carried_ours": {"points": a},
                                     "carried_ppg": {"points": b}}})
        summary = ledger.head_to_head("carried_ours", "carried_ppg")
        self.assertEqual(summary["paired_gameweeks"], 3)
        self.assertAlmostEqual(summary["mean_gain_per_gw"], (10 + 3 + 9) / 3.0)

    def test_head_to_head_only_uses_gameweeks_where_both_were_locked(self):
        """A week where one track was missing must not be borrowed from the other."""
        ledger = SeasonLedger("2026-27")
        ledger.add({"gameweek": 6, "ours": 58,
                    "variants": {"carried_ours": {"points": 60},
                                 "carried_ppg": {"points": 50}}})
        ledger.add({"gameweek": 7, "ours": 58,
                    "variants": {"carried_ours": {"points": 99}}})
        self.assertEqual(
            ledger.head_to_head("carried_ours", "carried_ppg")["paired_gameweeks"], 1)


if __name__ == "__main__":
    unittest.main()
