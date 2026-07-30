"""GW1 dry-run: prove the cold-start path before it is load-bearing on the real day.

The whole pipeline has one gameweek a year it has never actually run on real data,
and it is the one that cannot be re-tried: GW1. FPL zeroes every season-to-date stat
between seasons, so at GW1 every player's current-season rate is 0. Fed that raw, the
projection collapses — a Haaland-shaped striker and a bench striker both project the
appearance-points floor and nothing else, because every rate that separates them is
zero. Measured earlier: 300 players produced THREE distinct projections. A squad
picked from that is essentially random among same-position players.

The cold-start (docs/05, "Option C") is the fix: for the opening weeks, a player with
no minutes this season inherits last season's rates and ppg. This rehearses that fix
end to end on a synthetic rollover — a zeroed "new season" plus a differentiated
"last season" — and asserts the property that actually matters:

    WITHOUT cold-start, GW1 projections collapse to near-identical values.
    WITH cold-start, they are differentiated again, and the squad is legal and sane.

If this ever fails, the live GW1 squad would be picked blind, and we would not find
out until after the deadline — the one moment it is unrecoverable. So it is pinned
here, runs on fixtures, and needs no lake, no network and no model fit.
"""

import json
import unittest

import numpy as np
import pandas as pd

from prediction_engine.fpl import bank_it, carried, scorer
from prediction_engine.fpl.optimizer import select_squad
from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID

TEAMS = ["Arsenal", "Chelsea", "Liverpool", "Everton", "Brighton", "Fulham"]


class _StubEngine:
    """A scoreline grid and the match history `team_scoring_rates` reads."""

    def __init__(self, teams):
        rows = []
        for i, home in enumerate(teams):
            for away in teams:
                if home != away:
                    rows.append({"season": "2026-27", "home_team": home,
                                 "away_team": away,
                                 "home_goals": 2 if i % 2 else 1, "away_goals": 1})
        self.matches = pd.DataFrame(rows)

    def scoreline_grid(self, home, away, allow_unseen=False):
        grid = np.array([[0.06, 0.05, 0.02, 0.01, 0.00, 0.00],
                         [0.09, 0.08, 0.03, 0.01, 0.00, 0.00],
                         [0.08, 0.07, 0.03, 0.01, 0.00, 0.00],
                         [0.05, 0.04, 0.02, 0.01, 0.00, 0.00],
                         [0.02, 0.02, 0.01, 0.00, 0.00, 0.00],
                         [0.01, 0.01, 0.00, 0.00, 0.00, 0.00]])
        return grid / grid.sum()


def _zeroed_new_season_league(teams):
    """A GW1 FPL player table as FPL actually serves it: every season-to-date stat
    wiped to zero. What separates players is entirely gone until cold-start restores it."""
    rows, pid = [], 0
    for team in teams:
        for position in (GKP, GKP, DEF, DEF, MID, MID, FWD, FWD):
            pid += 1
            rows.append({
                "id": pid, "code": 500000 + pid,
                "web_name": "P%d" % pid, "full_name": "Player %d" % pid,
                "team": team, "position": position,
                "price": 4.0 + (pid % 7) * 0.5,
                # The wipe: no minutes, no points, no rates. All zero.
                "minutes": 0, "starts": 0, "total_points": 0,
                "goals_scored": 0, "assists": 0, "saves": 0, "bonus": 0,
                "yellow_cards": 0, "red_cards": 0, "defensive_contribution": 0,
                "expected_goals": 0.0, "expected_assists": 0.0,
                "available": True, "chance_of_playing": 100.0,
                "xg_per_90": 0.0, "xa_per_90": 0.0, "saves_per_90": 0.0,
                "bonus_per_90": 0.0, "dc_per_90": 0.0, "cards_per_90": 0.0,
            })
    return pd.DataFrame(rows)


def _last_season_rates(players):
    """Differentiated prior rates: within each position, a clear spread from elite to
    fringe, so a working cold-start MUST separate them."""
    rates = {}
    for _, row in players.iterrows():
        pid, position = int(row["id"]), int(row["position"])
        tier = (pid % 5) / 4.0                       # 0.0 (fringe) .. 1.0 (elite)
        rates[pid] = {
            "xg_per_90": (0.15 + 0.6 * tier) if position == FWD else
                         (0.10 + 0.3 * tier) if position == MID else 0.03,
            "xa_per_90": (0.10 + 0.4 * tier) if position in (MID, FWD) else 0.05,
            "saves_per_90": (2.0 + 2.0 * tier) if position == GKP else 0.0,
            "bonus_per_90": 0.1 + 0.4 * tier,
            "dc_per_90": (8.0 + 6.0 * tier) if position == DEF else 2.0 + tier,
            "cards_per_90": 0.1,
        }
    return rates


def _last_season_ppg(players):
    return {int(row["id"]): 2.0 + (int(row["id"]) % 5) * 1.4
            for _, row in players.iterrows()}


def _last_season_minutes(players):
    """Last season's per-gameweek minutes, differentiated by the same tier as the
    rates: an elite/nailed player ended the season on ~90 a game, a fringe player on
    cameos. This is what the minutes cold-start feeds the recent-form model at GW1."""
    history = {}
    for _, row in players.iterrows():
        pid = int(row["id"])
        tier = (pid % 5) / 4.0                       # 0.0 fringe .. 1.0 nailed
        per_game = int(round(20 + 70 * tier))        # 20 (cameo) .. 90 (nailed)
        history[pid] = [per_game] * 10               # a stable last-season sequence
    return history


class TestGw1ColdStart(unittest.TestCase):
    def setUp(self):
        # Isolate the carried-track state dir: these tests advance the tracks, and must
        # never read or write the real research/results/live/ (a stray dry-run there
        # once leaked a real squad's ids into this synthetic test).
        import tempfile
        from pathlib import Path
        from prediction_engine.fpl import carried
        self._real_live_dir = carried.LIVE_DIR
        carried.LIVE_DIR = Path(tempfile.mkdtemp())

        self.engine = _StubEngine(TEAMS)
        self.players = _zeroed_new_season_league(TEAMS)
        self.fixtures = [("Arsenal", "Chelsea"), ("Liverpool", "Everton"),
                         ("Brighton", "Fulham")]
        self.prior_rates = _last_season_rates(self.players)
        self.prior_ppg = _last_season_ppg(self.players)
        self.prior_minutes = _last_season_minutes(self.players)
        # At GW1 there is no current-season minutes history at all: FPL wiped it.
        self.minutes_history = {}

    def tearDown(self):
        import shutil
        from prediction_engine.fpl import carried
        shutil.rmtree(str(carried.LIVE_DIR), ignore_errors=True)
        carried.LIVE_DIR = self._real_live_dir

    def _frame(self, players, minutes_history="crude"):
        # Default "crude" mirrors the raw GW1 state (no history -> crude flat average
        # of zero minutes). Passing an explicit history rehearses the cold-start fix.
        history = self.minutes_history or None if minutes_history == "crude" else minutes_history
        return bank_it.build_gameweek_frame(
            self.engine, players, self.fixtures, minutes_history=history)

    def _coldstarted(self):
        """The players and minutes history exactly as `bank_gameweek` assembles them
        for an opening week: rates from last season, and minutes from last season for
        everyone who has not featured yet (all of them, at GW1)."""
        cold_players = bank_it.apply_early_season_rates(self.players, self.prior_rates)
        history = bank_it.coldstart_minutes_history(
            self.players, self.minutes_history, self.prior_minutes)
        return cold_players, history

    def test_raw_gw1_projections_collapse_to_zero(self):
        """The bug this all exists to prevent — demonstrated, so the fix has something
        to beat. With FPL's between-season wipe, BOTH rates and minutes are zero, so
        every projection is the zero the rates get multiplied into."""
        frame = self._frame(self.players)
        self.assertEqual(frame["expected_points"].nunique(), 1,
                         "raw GW1 (zero rates AND zero minutes) should give one value")
        self.assertAlmostEqual(frame["expected_points"].max(), 0.0,
                               msg="a squad picked from this is arbitrary")

    def test_rates_coldstart_alone_is_not_enough_without_minutes(self):
        """The subtle trap the dry-run caught: cold-starting the RATES but leaving
        minutes at the crude zero still projects everyone at zero, because the rates
        are multiplied by expected minutes. This pins WHY minutes must be cold-started
        too — so a future refactor cannot quietly drop the minutes half."""
        cold = bank_it.apply_early_season_rates(self.players, self.prior_rates)
        frame = self._frame(cold)                    # rates cold-started, minutes crude(0)
        self.assertAlmostEqual(frame["expected_points"].max(), 0.0,
                               msg="rates without minutes is still zero everywhere")

    def test_full_coldstart_restores_differentiation(self):
        cold, history = self._coldstarted()
        frame = self._frame(cold, minutes_history=history)
        distinct = frame.groupby("position")["expected_points"].nunique()
        self.assertGreater(int(distinct.min()), 3,
                           "the full cold-start must separate players again")
        self.assertGreater(frame["expected_points"].max(), 2.0,
                           "nailed players should project real points, not the floor")

        # Separation in the RIGHT direction: the nailed elite forward out-projects the
        # fringe one, on BOTH minutes and rate.
        forwards = frame[frame["position"] == FWD].set_index("player_id")
        elite = [pid for pid in forwards.index if pid % 5 == 4]
        fringe = [pid for pid in forwards.index if pid % 5 == 0]
        self.assertGreater(forwards.loc[elite, "expected_points"].mean(),
                           forwards.loc[fringe, "expected_points"].mean(),
                           "cold-start should rank last season's elite above the fringe")

    def test_recent_minutes_history_reads_the_current_season_not_the_last_ingested(self):
        """The id-reassignment guard. `_recent_minutes_history` must query the CURRENT
        season, not `ALL_SEASONS[-1]` (the last INGESTED one). At GW1 of a new season
        those differ: the archive still ends at last season, whose element ids have
        been reassigned, so keying this season's lookups by them hands each player a
        different player's minutes (measured: Saka got a benchwarmer's). Reading the
        current season returns {} until its data exists, letting the code-joined
        cold-start take over."""
        from prediction_engine.fpl import cli

        asked = {}
        real_load = cli.load_gameweeks
        real_season = None
        try:
            import research.data.predicted_lineups as pl
            real_season = pl.current_season
            pl.current_season = lambda *a, **k: "2026-27"

            def fake_load(season):
                asked["season"] = season
                raise FileNotFoundError("2026-27 not ingested yet")
            cli.load_gameweeks = fake_load

            result = cli._recent_minutes_history()
            self.assertEqual(asked["season"], "2026-27",
                             "must query the current season, not the last ingested one")
            self.assertEqual(result, {},
                             "no current-season data yet -> empty, so cold-start fires")
        finally:
            cli.load_gameweeks = real_load
            if real_season is not None:
                pl.current_season = real_season

    def test_minutes_coldstart_only_fills_players_without_current_history(self):
        """The prior must fade out: a player who has already featured this season keeps
        his own minutes, so the cold-start does not overwrite live form."""
        current = {1: [90, 90, 90]}                  # player 1 has featured this season
        merged = bank_it.coldstart_minutes_history(self.players, current,
                                                   self.prior_minutes)
        self.assertEqual(merged[1], [90, 90, 90], "live history must win over the prior")
        self.assertEqual(merged[2], self.prior_minutes[2], "player 2 inherits the prior")

    def test_the_whole_gw1_chain_runs_and_locks_a_legal_squad(self):
        """projection -> cold-start -> squad + baseline + carried tracks -> artifact
        -> JSON round-trip -> score -> ledger, all at GW1."""
        cold, history = self._coldstarted()
        frame = self._frame(cold, minutes_history=history)

        squad = bank_it.pick_squad(frame, budget=100.0)
        self.assertIsNotNone(squad, "a legal GW1 squad must exist")

        # Baseline uses last season's ppg at GW1 (the pre-registered opening-week rule).
        frame["player_ppg"] = frame["player_id"].map(
            bank_it.baseline_ppg(cold, self.minutes_history or None,
                                 prior_ppg=self.prior_ppg, gameweek=1))
        self.assertGreater(frame["player_ppg"].nunique(), 3,
                           "the GW1 baseline must not be all-zero (Option C)")
        baseline = select_squad(frame, "player_ppg", squad_budget=100.0)
        self.assertIsNotNone(baseline)

        # Carried tracks open at GW1 from the primary's fifteen.
        opening_ids = [int(frame.loc[idx, "player_id"])
                       for idx in list(squad.xi) + list(squad.bench)]
        blocks = carried.advance_tracks(frame, opening_ids, "2026-27", 1, persist=False)
        self.assertEqual(set(blocks), {"carried_ours", "carried_ppg"})
        for block in blocks.values():
            self.assertEqual(len(block["xi"]), 11)

        artifact = bank_it.build_artifact(
            frame, squad, gameweek=1, deadline="2026-08-21T17:30:00Z",
            config={"budget": 100.0, "scoring_status": "EXPLORATORY"},
            baseline_squad=baseline, extra_variants=blocks)
        artifact = json.loads(json.dumps(artifact))         # the real on-disk round-trip

        self.assertEqual(len(artifact["xi"]), 11)
        self.assertEqual(len(artifact["bench"]), 4)
        self.assertEqual(sum(1 for r in artifact["xi"] if r["captain"]), 1)

        # GW1 is exploratory (outside the validated GW6-38 range) — it must be scored
        # and reported, but never folded into the pre-registered primary.
        actuals = {int(p): {"points": 2.0 + (int(p) % 9), "minutes": 90.0}
                   for p in self.players["id"]}
        record = scorer.score_artifact(artifact, actuals)
        ledger = scorer.SeasonLedger("2026-27")
        ledger.add(record)
        summary = ledger.summary()
        self.assertEqual(summary["scored_gameweeks"], 0,
                         "GW1 must NOT count toward the validated primary")
        self.assertEqual(summary["exploratory"]["gameweeks"], 1,
                         "GW1 must be reported as exploratory")

    def test_squad_obeys_every_fpl_rule_at_gw1(self):
        cold, history = self._coldstarted()
        frame = self._frame(cold, minutes_history=history)
        squad = bank_it.pick_squad(frame, budget=100.0)
        picked = frame.loc[list(squad.xi) + list(squad.bench)]
        self.assertEqual(len(picked), 15)
        counts = picked["position"].value_counts().to_dict()
        self.assertEqual({counts.get(1), counts.get(2), counts.get(3), counts.get(4)},
                         {2, 5, 5, 3}, "squad quota 2/5/5/3")
        self.assertLessEqual(picked["price"].sum(), 100.0 + 1e-9)
        self.assertLessEqual(picked["club"].value_counts().max(), 3)


if __name__ == "__main__":
    unittest.main()
