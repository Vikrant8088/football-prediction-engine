"""The forward-validation driver: score a locked artifact, accumulate the ledger.

This is orchestration, so the tests inject actuals rather than hit the network. What
they pin is that the driver joins the pieces without losing either integrity property
it inherits: it scores the LOCKED squad (never a rebuilt one), and re-running a
gameweek overwrites its ledger row rather than double-counting it.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from prediction_engine.fpl import score_live


def _artifact(gameweek):
    """A minimal locked artifact with a primary, a baseline, and a variant."""
    return {
        "gameweek": gameweek,
        "captain_id": 1,
        "xi": [{"player_id": pid} for pid in range(1, 12)],
        "baseline": {"xi": list(range(12, 23)), "captain_id": 12},
        "variants": {
            "carried_ours": {"xi": list(range(1, 12)), "captain_id": 1, "hits": 0},
            "carried_ppg": {"xi": list(range(12, 23)), "captain_id": 12, "hits": 0},
        },
    }


class _TempLive(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        (self.base / "2026-27").mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(str(self.base), ignore_errors=True)

    def _write_artifact(self, gameweek):
        path = self.base / "2026-27" / ("GW%02d.json" % gameweek)
        path.write_text(json.dumps(_artifact(gameweek)), encoding="utf-8")


class TestScoringDriver(_TempLive):
    def test_it_scores_a_locked_gameweek_and_writes_the_ledger(self):
        self._write_artifact(6)
        actuals = {pid: {"points": 5.0, "minutes": 90.0} for pid in range(1, 23)}
        result = score_live.score_gameweek("2026-27", 6, actuals=actuals, base=self.base)

        # XI of 11 all on 5, captain (id 1) doubled -> 11*5 + 5 = 60.
        self.assertEqual(result["record"]["ours"], 60.0)
        self.assertTrue(score_live.ledger_path("2026-27", self.base).exists())

    def test_rescoring_a_gameweek_overwrites_rather_than_duplicates(self):
        self._write_artifact(6)
        good = {pid: {"points": 5.0, "minutes": 90.0} for pid in range(1, 23)}
        score_live.score_gameweek("2026-27", 6, actuals=good, base=self.base)
        # Re-run with corrected actuals (e.g. a bonus adjustment).
        fixed = {pid: {"points": 6.0, "minutes": 90.0} for pid in range(1, 23)}
        result = score_live.score_gameweek("2026-27", 6, actuals=fixed, base=self.base)

        ledger = score_live.load_ledger("2026-27", self.base)
        self.assertEqual(len(ledger.records), 1, "one gameweek, not two")
        self.assertEqual(result["record"]["ours"], 72.0)

    def test_a_missing_artifact_is_a_clear_error_not_a_silent_zero(self):
        with self.assertRaises(FileNotFoundError):
            score_live.score_gameweek("2026-27", 9,
                                      actuals={1: {"points": 1.0, "minutes": 90.0}},
                                      base=self.base)

    def test_the_running_summary_accumulates_across_gameweeks(self):
        actuals = {pid: {"points": 4.0, "minutes": 90.0} for pid in range(1, 23)}
        for gameweek in (6, 7, 8):
            self._write_artifact(gameweek)
            result = score_live.score_gameweek("2026-27", gameweek, actuals=actuals,
                                               base=self.base)
        self.assertEqual(result["summary"]["scored_gameweeks"], 3)

    def test_the_carried_head_to_head_appears_in_the_summary(self):
        # carried_ours (ids 1-11, cap 1) and carried_ppg (ids 12-22, cap 12) score
        # differently, so the A/B should register a nonzero paired comparison.
        for gameweek in (6, 7):
            self._write_artifact(gameweek)
            actuals = {pid: {"points": (9.0 if pid <= 11 else 3.0), "minutes": 90.0}
                       for pid in range(1, 23)}
            result = score_live.score_gameweek("2026-27", gameweek, actuals=actuals,
                                               base=self.base)
        h2h = result["summary"]["carried_head_to_head"]
        self.assertEqual(h2h["paired_gameweeks"], 2)
        self.assertGreater(h2h["mean_gain_per_gw"], 0)


if __name__ == "__main__":
    unittest.main()
