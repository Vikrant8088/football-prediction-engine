"""Tests for the Bank-It scorer and season ledger (docs/04_BANK_IT_PIPELINE.md §4.6).

Two kinds of check:
  - unit behaviour: captain doubling, artifact scoring, the paired test, the ledger;
  - a cross-check that `score_squad` reproduces the optimizer's already-validated
    `xi_actual_points` on real cached backtest data — so the live scorer measures
    the same thing the proven backtest does.
"""

import glob
import json
import os
import tempfile
import unittest

import pandas as pd

from prediction_engine.fpl import scorer


class TestScoreSquad(unittest.TestCase):
    def test_captain_is_counted_twice(self):
        actuals = {1: 5.0, 2: 3.0, 3: 2.0}
        # 5 + 3 + 2, plus the captain (id 1) again = 15.
        self.assertAlmostEqual(scorer.score_squad(actuals, [1, 2, 3], captain_id=1), 15.0)

    def test_absent_player_scores_zero(self):
        actuals = {1: 5.0}                        # ids 2, 3 did not feature
        self.assertAlmostEqual(scorer.score_squad(actuals, [1, 2, 3], captain_id=2), 5.0)


class TestScoreArtifact(unittest.TestCase):
    def _artifact(self):
        return {
            "gameweek": 7,
            "captain_id": 10,
            "xi": [{"player_id": 10}, {"player_id": 11}, {"player_id": 12}],
            "baseline": {"xi": [20, 21, 12], "captain_id": 20},
        }

    def test_scores_ours_baseline_and_gain(self):
        actuals = {10: 8.0, 11: 2.0, 12: 3.0, 20: 4.0, 21: 1.0}
        record = scorer.score_artifact(self._artifact(), actuals)
        self.assertAlmostEqual(record["ours"], 8 + 2 + 3 + 8)      # captain 10 doubled = 21
        self.assertAlmostEqual(record["baseline"], 4 + 1 + 3 + 4)  # captain 20 doubled = 12
        self.assertAlmostEqual(record["gain"], 21 - 12)

    def test_artifact_without_baseline_has_no_gain(self):
        artifact = {"gameweek": 1, "captain_id": 1, "xi": [{"player_id": 1}]}
        record = scorer.score_artifact(artifact, {1: 5.0})
        self.assertNotIn("gain", record)


class TestPairedSummary(unittest.TestCase):
    def test_matches_hand_computed_gain(self):
        ours = [10.0, 12.0, 8.0]
        baseline = [8.0, 9.0, 9.0]                # diffs: +2, +3, -1 -> mean 1.333, 2 wins
        summary = scorer.paired_summary(ours, baseline)
        self.assertAlmostEqual(summary["mean_gain_per_gw"], 4.0 / 3.0)
        self.assertEqual(summary["gameweeks_won"], 2)
        self.assertEqual(summary["gameweeks"], 3)

    def test_empty_is_safe(self):
        summary = scorer.paired_summary([], [])
        self.assertEqual(summary["gameweeks"], 0)
        self.assertFalse(summary["significant"])


class TestSeasonLedger(unittest.TestCase):
    def _records(self):
        return [
            {"gameweek": 1, "ours": 60.0, "baseline": 55.0, "gain": 5.0},
            {"gameweek": 2, "ours": 50.0, "baseline": 52.0, "gain": -2.0},
        ]

    def test_add_overwrites_same_gameweek(self):
        ledger = scorer.SeasonLedger("2026-27")
        ledger.add({"gameweek": 1, "ours": 60.0, "baseline": 55.0, "gain": 5.0})
        ledger.add({"gameweek": 1, "ours": 61.0, "baseline": 55.0, "gain": 6.0})
        self.assertEqual(len(ledger.records), 1)
        self.assertAlmostEqual(ledger.records[0]["ours"], 61.0)

    def test_summary_runs_the_paired_test(self):
        ledger = scorer.SeasonLedger("2026-27", self._records())
        summary = ledger.summary()
        self.assertEqual(summary["scored_gameweeks"], 2)
        self.assertAlmostEqual(summary["mean_gain_per_gw"], 1.5)   # (5 + -2)/2

    def test_save_and_load_round_trip(self):
        ledger = scorer.SeasonLedger("2026-27", self._records())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            ledger.save(path)
            reloaded = scorer.SeasonLedger.load(path)
        self.assertEqual(len(reloaded.records), 2)
        self.assertAlmostEqual(reloaded.summary()["mean_gain_per_gw"], 1.5)


def _cached_understat_frame():
    """The shipped backtest frame, if it has been computed, else None."""
    from data_warehouse.config.loader import load_config
    processed = load_config().raw_data_dir.parent / "processed"
    matches = sorted(glob.glob(str(processed / "fpl_backtest_predictions_understat*.csv")))
    return pd.read_csv(matches[0]) if matches else None


class TestScorerMatchesOptimizer(unittest.TestCase):
    """`score_squad` must agree with the proven `xi_actual_points` on real data."""

    def test_agrees_on_a_real_gameweek(self):
        frame = _cached_understat_frame()
        if frame is None:
            self.skipTest("no cached backtest frame on disk")
        from prediction_engine.fpl.optimizer import select_squad, xi_actual_points

        for _, week in frame.groupby(["season", "gameweek"]):
            week = week.reset_index(drop=True)
            squad = select_squad(week, "ours")
            if squad is None:
                continue
            actuals = {int(pid): float(pts)
                       for pid, pts in zip(week["player_id"], week["actual"])}
            xi_ids = [int(week.loc[idx, "player_id"]) for idx in squad.xi]
            captain_id = int(week.loc[squad.captain, "player_id"])

            mine = scorer.score_squad(actuals, xi_ids, captain_id)
            proven = xi_actual_points(week, squad, with_captain=True)
            self.assertAlmostEqual(mine, proven, places=6)
            return          # one real gameweek is enough to tie the two scorers
        self.skipTest("no legal squad found in cached frame")


if __name__ == "__main__":
    unittest.main()
