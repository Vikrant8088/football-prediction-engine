"""The season watcher must not cry wolf, and must not sleep through the rollover.

Both failure directions are expensive and neither is loud:

  false positive  a spurious alert trains us to ignore the alert, which is worse
                  than having none
  false negative  M3 and the data refresh silently do not start, and the first
                  anyone notices is a GW1 squad picked on last season's rules

so both get tested against fixtures, with no network.
"""

import unittest

from research.data import fpl_season_watch as watch


def _event(event_id, finished, is_next=False, deadline="2026-08-21T17:30:00Z"):
    return {"id": event_id, "deadline_time": deadline,
            "finished": finished, "is_next": is_next}


def _payload(events, teams=None, players=600):
    return {
        "events": events,
        "teams": [{"id": i + 1, "name": name}
                  for i, name in enumerate(teams or sorted(watch.KNOWN_TEAMS_2025_26))],
        "elements": [{"id": i} for i in range(players)],
    }


class TestBetweenSeasons(unittest.TestCase):
    """The state the repo is in right now: FPL still serving the finished season."""

    def test_a_fully_finished_season_is_not_a_rollover(self):
        payload = _payload([_event(i, finished=True) for i in range(1, 39)])
        status = watch.season_status(payload)
        self.assertFalse(status["rolled_over"])
        self.assertEqual(status["finished_events"], 38)

    def test_no_next_gameweek_is_reported_between_seasons(self):
        payload = _payload([_event(i, finished=True) for i in range(1, 39)])
        self.assertIsNone(watch.season_status(payload)["next_gameweek"])


class TestRollover(unittest.TestCase):
    def test_a_published_new_season_is_detected(self):
        payload = _payload([_event(1, finished=False, is_next=True)]
                           + [_event(i, finished=False) for i in range(2, 39)])
        status = watch.season_status(payload)
        self.assertTrue(status["rolled_over"])
        self.assertEqual(status["next_gameweek"], 1)
        self.assertEqual(status["next_deadline"], "2026-08-21T17:30:00Z")

    def test_it_detects_publication_before_the_is_next_flag_settles(self):
        """There is a window where the new events exist but none is flagged `is_next`.
        Waiting for the flag would sleep through it."""
        payload = _payload([_event(i, finished=False) for i in range(1, 39)])
        status = watch.season_status(payload)
        self.assertTrue(status["rolled_over"])
        self.assertEqual(status["next_gameweek"], 1)

    def test_mid_season_is_also_rolled_over(self):
        # Not the case we are waiting for, but the watcher must never report the
        # LIVE season as "not open" — that would be a false negative in the one
        # state where it matters most.
        payload = _payload([_event(i, finished=True) for i in range(1, 10)]
                           + [_event(10, finished=False, is_next=True)]
                           + [_event(i, finished=False) for i in range(11, 39)])
        status = watch.season_status(payload)
        self.assertTrue(status["rolled_over"])
        self.assertEqual(status["next_gameweek"], 10)


class TestPromotionDetail(unittest.TestCase):
    """Reported, never the trigger — the team list changes at a slightly different
    moment from the events, and a two-signal trigger is two things that can disagree."""

    def test_promoted_and_relegated_clubs_are_named(self):
        teams = (sorted(watch.KNOWN_TEAMS_2025_26 - {"Burnley", "Leeds", "Sunderland"})
                 + ["Birmingham", "Coventry", "Southampton"])
        payload = _payload([_event(1, finished=False, is_next=True)], teams=teams)
        status = watch.season_status(payload)
        self.assertEqual(status["promoted"], ["Birmingham", "Coventry", "Southampton"])
        self.assertEqual(status["relegated"], ["Burnley", "Leeds", "Sunderland"])

    def test_an_unchanged_team_list_does_not_trigger_a_rollover(self):
        payload = _payload([_event(i, finished=True) for i in range(1, 39)])
        status = watch.season_status(payload)
        self.assertFalse(status["rolled_over"])
        self.assertEqual(status["promoted"], [])


class TestAlertContent(unittest.TestCase):
    """The alert is built here rather than in the workflow's shell precisely so it can
    be tested. If it were assembled through YAML -> shell -> jq, none of this would be
    checkable until the day it fires — which is the one day it must not be wrong."""

    def _status(self):
        return watch.season_status(_payload(
            [_event(1, finished=False, is_next=True)],
            teams=(sorted(watch.KNOWN_TEAMS_2025_26 - {"Burnley"}) + ["Southampton"]),
            players=712))

    def test_the_title_names_the_gameweek_and_deadline(self):
        title = watch.issue_title(self._status())
        self.assertIn("GW1", title)
        self.assertIn("2026-08-21T17:30:00Z", title)

    def test_the_body_carries_the_facts_and_the_checklist(self):
        body = watch.issue_body(self._status())
        self.assertIn("**Players in the game:** 712", body)
        self.assertIn("Southampton", body)               # promoted, so cold-started
        self.assertIn("M3", body)
        self.assertIn("M5", body)
        self.assertEqual(body.count("- [ ]"), 5, "five blocked tasks")

    def test_no_promotions_detected_reads_as_a_sentence_not_an_empty_string(self):
        status = watch.season_status(_payload([_event(1, finished=False)]))
        self.assertIn("none detected", watch.issue_body(status))

    def test_the_alert_files_are_written_only_on_a_rollover(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path

        real = watch.fetch_bootstrap
        try:
            for rolled, expected in ((False, False), (True, True)):
                events = ([_event(1, finished=False, is_next=True)] if rolled
                          else [_event(i, finished=True) for i in range(1, 39)])
                watch.fetch_bootstrap = lambda url=None, e=events: _payload(e)
                out = Path(tempfile.mkdtemp())
                with contextlib.redirect_stdout(io.StringIO()):
                    watch.main(["--out", str(out)])
                self.assertTrue((out / "status.json").exists(), "status is always written")
                self.assertEqual((out / "issue_body.md").exists(), expected)
                self.assertEqual((out / "issue_title.txt").exists(), expected)
        finally:
            watch.fetch_bootstrap = real


class TestExitCode(unittest.TestCase):
    """The workflow branches on the exit code, so it is part of the contract."""

    def setUp(self):
        self._real = watch.fetch_bootstrap

    def tearDown(self):
        watch.fetch_bootstrap = self._real

    def _silent_main(self):
        """`main` prints the status payload, which is the point in CI and pure noise
        in a test run."""
        import contextlib
        import io
        # `[]`, not None: argparse would otherwise read sys.argv, which under a test
        # runner holds the runner's own arguments.
        with contextlib.redirect_stdout(io.StringIO()):
            return watch.main([])

    def test_waiting_exits_zero_and_rollover_exits_one(self):
        watch.fetch_bootstrap = lambda url=None: _payload(
            [_event(i, finished=True) for i in range(1, 39)])
        self.assertEqual(self._silent_main(), 0)

        watch.fetch_bootstrap = lambda url=None: _payload(
            [_event(1, finished=False, is_next=True)])
        self.assertEqual(self._silent_main(), 1)


if __name__ == "__main__":
    unittest.main()
