"""Same-week live scoring, and the refusal that keeps it honest.

The one thing this module must not do is score a gameweek whose bonus is still
provisional: those points change when FPL confirms them, and a ledger built on
numbers that later move is quietly wrong in the same family as hindsight. So the
`data_checked` refusal is the load-bearing test here, alongside the shape contract
that lets it stand in for the archive with no special case.
"""

import unittest

from research.data import fpl_live


def _bootstrap(events):
    return {"events": events}


def _event(event_id, finished=True, data_checked=True):
    return {"id": event_id, "finished": finished, "data_checked": data_checked}


def _live(elements):
    return {"elements": elements}


def _element(element_id, points, minutes):
    return {"id": element_id,
            "stats": {"total_points": points, "minutes": minutes}}


class TestParsing(unittest.TestCase):
    def test_it_returns_the_scorer_shape(self):
        payload = _live([_element(1, 12, 90), _element(2, 2, 45)])
        actuals = fpl_live.parse_live(payload)
        self.assertEqual(actuals[1], {"points": 12.0, "minutes": 90.0})
        self.assertEqual(actuals[2], {"points": 2.0, "minutes": 45.0})

    def test_missing_stats_do_not_crash(self):
        # A player who did not feature may carry zeros rather than a full stats block.
        payload = {"elements": [{"id": 7, "stats": {}}]}
        self.assertEqual(fpl_live.parse_live(payload)[7], {"points": 0.0, "minutes": 0.0})

    def test_keys_are_integers(self):
        actuals = fpl_live.parse_live(_live([_element(5, 6, 90)]))
        self.assertIn(5, actuals)
        self.assertNotIn("5", actuals)


class TestFinality(unittest.TestCase):
    def test_data_checked_is_the_scoreable_signal(self):
        b = _bootstrap([_event(20, finished=True, data_checked=True)])
        self.assertTrue(fpl_live.gameweek_finality(b, 20)["scoreable"])

    def test_finished_but_not_data_checked_is_not_scoreable(self):
        # The dangerous window: matches are over but bonus is still provisional.
        b = _bootstrap([_event(20, finished=True, data_checked=False)])
        finality = fpl_live.gameweek_finality(b, 20)
        self.assertTrue(finality["finished"])
        self.assertFalse(finality["scoreable"])

    def test_an_unpublished_gameweek_reports_not_exists(self):
        self.assertFalse(fpl_live.gameweek_finality(_bootstrap([]), 20)["exists"])


class TestLiveActualsGuard(unittest.TestCase):
    def setUp(self):
        self._boot, self._live = fpl_live.fetch_bootstrap, fpl_live.fetch_live

    def tearDown(self):
        fpl_live.fetch_bootstrap, fpl_live.fetch_live = self._boot, self._live

    def _patch(self, events, elements):
        fpl_live.fetch_bootstrap = lambda: _bootstrap(events)
        fpl_live.fetch_live = lambda gw: _live(elements)

    def test_it_refuses_to_score_a_gameweek_that_is_not_final(self):
        self._patch([_event(20, finished=True, data_checked=False)],
                    [_element(1, 12, 90)])
        with self.assertRaises(fpl_live.LiveScoringError):
            fpl_live.live_actuals(20)

    def test_it_refuses_an_unpublished_gameweek(self):
        self._patch([], [_element(1, 12, 90)])
        with self.assertRaises(fpl_live.LiveScoringError):
            fpl_live.live_actuals(20)

    def test_it_scores_a_final_gameweek(self):
        self._patch([_event(20, finished=True, data_checked=True)],
                    [_element(1, 12, 90), _element(2, 0, 0)])
        actuals = fpl_live.live_actuals(20)
        self.assertEqual(actuals[1]["points"], 12.0)

    def test_a_provisional_read_is_allowed_only_when_explicitly_requested(self):
        self._patch([_event(20, finished=True, data_checked=False)],
                    [_element(1, 12, 90)])
        # Opt-in provisional read returns numbers instead of raising.
        actuals = fpl_live.live_actuals(20, require_final=False)
        self.assertEqual(actuals[1]["points"], 12.0)


if __name__ == "__main__":
    unittest.main()
