"""The carried-squad manager: rules first, policy second.

The rules tested here are the ones that decide whether a measured "transfers help"
result is real or an artifact. Nearly every way to get a fake edge in a transfer
backtest is a rule bent in the manager's favour:

  - refund the full current price on a sale and he is handed free money every week
  - let free transfers bank without limit and he never takes a hit
  - forget the 3-per-club cap or the position quota and he buys an illegal team
  - judge a transfer on the two players rather than on the XI and he "upgrades" a
    bench player who still does not start

so each of those gets a test that fails if the rule is loosened.
"""

import unittest

import pandas as pd

from prediction_engine.fpl.manager import (DEFAULT_FREE_TRANSFER_CAP, HIT_COST,
                                           Squad, best_single_transfer, best_xi,
                                           execute, free_transfer_cap,
                                           opening_squad, plan_transfers,
                                           score_gameweek, selling_price_tenths,
                                           squad_projection)
from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID

# A legal 15: 2/5/5/3 spread over enough clubs to respect the 3-per-club cap.
POSITIONS_15 = [GKP, GKP] + [DEF] * 5 + [MID] * 5 + [FWD] * 3


def _squad_world(projections=None):
    """Ids 1..15 are the squad; 101.. are transfer targets."""
    positions = {i + 1: POSITIONS_15[i] for i in range(15)}
    clubs = {i + 1: "C%d" % (i % 6) for i in range(15)}
    prices = {i + 1: 50 for i in range(15)}
    proj = {i + 1: float(i + 1) for i in range(15)}
    if projections:
        proj.update(projections)
    return positions, clubs, prices, proj


class TestSellingPrice(unittest.TestCase):
    """FPL halves your profit and rounds it DOWN, and gives you none of it back on a
    fall. Refunding the current price instead is free money, every week, for every
    policy that transfers — and `hold` would get none of it."""

    def test_profit_is_halved_and_rounded_down(self):
        self.assertEqual(selling_price_tenths(70, 75), 72)   # +0.5 rise -> +0.2
        self.assertEqual(selling_price_tenths(70, 73), 71)   # +0.3 rise -> +0.1
        self.assertEqual(selling_price_tenths(70, 71), 70)   # +0.1 rise -> nothing

    def test_a_fall_is_borne_in_full(self):
        self.assertEqual(selling_price_tenths(70, 65), 65)

    def test_an_unchanged_price_returns_exactly_what_was_paid(self):
        self.assertEqual(selling_price_tenths(70, 70), 70)

    def test_selling_never_pays_more_than_the_market_price(self):
        for bought in range(38, 140):
            for current in range(38, 140):
                self.assertLessEqual(selling_price_tenths(bought, current), current)


class TestFreeTransferRules(unittest.TestCase):
    def test_the_cap_changed_in_2024_25_and_the_backtest_must_know(self):
        # Applying today's cap of 5 to 2022/23 measures a game nobody played.
        self.assertEqual(free_transfer_cap("2022-23"), 2)
        self.assertEqual(free_transfer_cap("2023-24"), 2)
        self.assertEqual(free_transfer_cap("2024-25"), 5)
        self.assertEqual(free_transfer_cap("2026-27"), 5)

    def test_an_unknown_season_falls_back_to_the_current_rule(self):
        self.assertEqual(free_transfer_cap("2031-32"), DEFAULT_FREE_TRANSFER_CAP)

    def test_free_transfers_bank_but_never_exceed_the_cap(self):
        squad = Squad([], {}, free_transfers=1, season="2022-23")
        squad.award_free_transfer()
        self.assertEqual(squad.free_transfers, 2)
        squad.award_free_transfer()
        self.assertEqual(squad.free_transfers, 2, "2022/23 capped banked transfers at 2")

        modern = Squad([], {}, free_transfers=1, season="2025-26")
        for _ in range(10):
            modern.award_free_transfer()
        self.assertEqual(modern.free_transfers, 5)

    def test_using_transfers_draws_the_balance_down_and_never_below_zero(self):
        squad = Squad([1], {1: 50}, free_transfers=1, season="2025-26")
        plan = type("P", (), {"moves": [{"out": 1, "in": 2, "in_price": 50,
                                         "out_price": 50}], "hits": HIT_COST})()
        execute(squad, plan)
        self.assertEqual(squad.free_transfers, 0)


class TestBestXi(unittest.TestCase):
    def test_it_returns_a_legal_formation_and_doubles_the_captain(self):
        positions, _, _, proj = _squad_world()
        choice = best_xi(list(range(1, 16)), proj, positions)
        self.assertEqual(len(choice.xi), 11)
        self.assertEqual(len(choice.bench), 4)

        counts = {}
        for pid in choice.xi:
            counts[positions[pid]] = counts.get(positions[pid], 0) + 1
        self.assertEqual(counts.get(GKP), 1)
        self.assertGreaterEqual(counts.get(DEF, 0), 3)
        self.assertGreaterEqual(counts.get(MID, 0), 2)
        self.assertGreaterEqual(counts.get(FWD, 0), 1)

        expected = sum(proj[pid] for pid in choice.xi) + max(proj[pid] for pid in choice.xi)
        self.assertAlmostEqual(choice.projected, expected)

    def test_it_beats_or_matches_every_legal_formation_by_brute_force(self):
        """The claim is that top-k-per-position over eight shapes is EXACT. Checked
        against explicit enumeration rather than asserted."""
        import itertools
        positions, _, _, proj = _squad_world()
        squad = list(range(1, 16))
        best_seen = 0.0
        for combo in itertools.combinations(squad, 11):
            counts = {}
            for pid in combo:
                counts[positions[pid]] = counts.get(positions[pid], 0) + 1
            if counts.get(GKP, 0) != 1 or not (3 <= counts.get(DEF, 0) <= 5):
                continue
            if not (2 <= counts.get(MID, 0) <= 5) or not (1 <= counts.get(FWD, 0) <= 3):
                continue
            total = sum(proj[p] for p in combo) + max(proj[p] for p in combo)
            best_seen = max(best_seen, total)
        self.assertAlmostEqual(best_xi(squad, proj, positions).projected, best_seen)

    def test_a_squad_that_cannot_field_a_legal_xi_returns_none(self):
        positions = {1: GKP, 2: GKP}
        self.assertIsNone(best_xi([1, 2], {1: 5.0, 2: 4.0}, positions))


class TestTransferSearch(unittest.TestCase):
    def test_it_values_the_xi_not_the_two_players(self):
        """The trap: a huge upgrade to a player who STILL will not start is worth
        nothing. A policy that scores transfers by the players' own projections buys
        it anyway, with real money."""
        positions, clubs, prices, proj = _squad_world()
        # Player 1 is a bench keeper (the squad's two keepers are 1 and 2; 2 is better).
        positions[101], clubs[101], prices[101], proj[101] = GKP, "C9", 50, 1.5
        proj[1], proj[2] = 1.0, 9.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, season="2025-26")
        before = squad_projection(squad.players, proj, positions)
        move = best_single_transfer(squad, proj, prices, clubs, positions, [101])
        # Swapping bench keeper 1 (1.0) for a better bench keeper (1.5) cannot change
        # the XI, so it must be reported as no gain at all.
        self.assertIsNone(move)
        self.assertAlmostEqual(before, squad_projection(squad.players, proj, positions))

    def test_it_finds_a_genuine_upgrade_to_a_starter(self):
        positions, clubs, prices, proj = _squad_world()
        positions[101], clubs[101], prices[101], proj[101] = FWD, "C9", 50, 99.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, season="2025-26")
        move = best_single_transfer(squad, proj, prices, clubs, positions, [101])
        self.assertIsNotNone(move)
        self.assertEqual(move["in"], 101)
        self.assertGreater(move["gain"], 0)

    def test_it_never_buys_a_player_it_cannot_afford(self):
        positions, clubs, prices, proj = _squad_world()
        positions[101], clubs[101], prices[101], proj[101] = FWD, "C9", 200, 99.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, season="2025-26")
        self.assertIsNone(best_single_transfer(squad, proj, prices, clubs, positions, [101]))

    def test_it_respects_the_three_per_club_limit(self):
        positions, clubs, prices, proj = _squad_world()
        # Fill club "FULL" with three squad players, then offer a fourth.
        for pid in (13, 14, 15):
            clubs[pid] = "FULL"
        positions[101], clubs[101], prices[101], proj[101] = FWD, "FULL", 50, 99.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, season="2025-26")
        move = best_single_transfer(squad, proj, prices, clubs, positions, [101])
        # Selling a FULL player to buy another FULL player is legal (count unchanged);
        # what must never happen is a fourth arriving while three remain.
        if move is not None:
            self.assertIn(move["out"], (13, 14, 15))

    def test_a_transfer_is_position_for_position(self):
        positions, clubs, prices, proj = _squad_world()
        positions[101], clubs[101], prices[101], proj[101] = DEF, "C9", 50, 99.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, season="2025-26")
        move = best_single_transfer(squad, proj, prices, clubs, positions, [101])
        self.assertIsNotNone(move)
        self.assertEqual(positions[move["out"]], DEF, "the quota 2/5/5/3 is fixed")

    def test_money_moves_correctly_through_a_transfer(self):
        squad = Squad([1], {1: 70}, bank_tenths=5, season="2025-26")
        # Sell at 75 having paid 70 -> receive 72, not 75. Buy at 60.
        squad.apply_transfer(1, 2, in_price_tenths=60, out_price_tenths=75)
        self.assertEqual(squad.bank, 5 + 72 - 60)
        self.assertEqual(squad.players, [2])
        self.assertEqual(squad.bought[2], 60)


class TestTransferSearchIsExact(unittest.TestCase):
    """`best_single_transfer` skips almost every candidate, on the argument that for a
    fixed player sold, the best incoming player is simply the highest-projected legal
    affordable one. That is a real claim about the search and it is worth what the
    optimizer's equivalent claim is worth: nothing until checked against brute force.

    So: random squads, random prices, random club spreads, tight budgets — and the
    fast answer compared against evaluating every legal (out, in) pair.
    """

    def _brute_force(self, squad, proj, prices, clubs, positions, candidates):
        baseline = squad_projection(squad.players, proj, positions)
        best_gain = 0.0
        held = set(squad.players)
        counts = squad.club_counts(clubs)
        for out_id in squad.players:
            out_price = prices.get(out_id, squad.bought.get(out_id, 0))
            budget = squad.bank + selling_price_tenths(
                squad.bought.get(out_id, out_price), out_price)
            for in_id in candidates:
                if in_id in held or positions.get(in_id) != positions.get(out_id):
                    continue
                if prices.get(in_id, 10 ** 9) > budget:
                    continue
                club = clubs.get(in_id)
                used = counts.get(club, 0) - (1 if club == clubs.get(out_id) else 0)
                if used >= 3:
                    continue
                remaining = [p for p in squad.players if p != out_id]
                gain = squad_projection(remaining + [in_id], proj, positions) - baseline
                best_gain = max(best_gain, gain)
        return best_gain

    def test_it_matches_exhaustive_search_on_random_instances(self):
        import random
        rng = random.Random(20260720)
        for trial in range(60):
            positions, clubs, prices, proj = _squad_world()
            for pid in range(1, 16):
                proj[pid] = round(rng.uniform(0.0, 9.0), 3)
                prices[pid] = rng.randint(40, 90)
                clubs[pid] = "C%d" % rng.randrange(7)
            candidates = []
            for offset in range(40):
                pid = 200 + offset
                positions[pid] = rng.choice([GKP, DEF, MID, FWD])
                clubs[pid] = "C%d" % rng.randrange(7)
                prices[pid] = rng.randint(40, 110)
                proj[pid] = round(rng.uniform(0.0, 11.0), 3)
                candidates.append(pid)

            squad = Squad(list(range(1, 16)), {p: prices[p] for p in range(1, 16)},
                          bank_tenths=rng.randint(0, 30), season="2025-26")
            # The squad is randomised, so it may itself breach the club cap; skip
            # those rather than test the search on a position it could never reach.
            if max(squad.club_counts(clubs).values()) > 3:
                continue

            fast = best_single_transfer(squad, proj, prices, clubs, positions, candidates)
            expected = self._brute_force(squad, proj, prices, clubs, positions, candidates)
            actual = 0.0 if fast is None else fast["gain"]
            self.assertAlmostEqual(
                actual, expected, places=9,
                msg="trial %d: fast search found %.6f, exhaustive found %.6f"
                    % (trial, actual, expected))


class TestPolicyThresholds(unittest.TestCase):
    def _world(self):
        positions, clubs, prices, proj = _squad_world()
        positions[101], clubs[101], prices[101], proj[101] = FWD, "C9", 50, 20.0
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)},
                      bank_tenths=0, free_transfers=1, season="2025-26")
        return squad, proj, prices, clubs, positions

    def test_hits_are_forbidden_by_default(self):
        squad, proj, prices, clubs, positions = self._world()
        positions[102], clubs[102], prices[102], proj[102] = FWD, "C8", 50, 19.0
        plan = plan_transfers(squad, proj, prices, clubs, positions, [101, 102],
                              max_transfers=2)
        self.assertEqual(plan.hits, 0, "the default policy must never take a hit")
        self.assertLessEqual(len(plan.moves), 1)

    def test_a_hit_is_taken_only_when_it_clears_the_threshold(self):
        squad, proj, prices, clubs, positions = self._world()
        positions[102], clubs[102], prices[102], proj[102] = FWD, "C8", 50, 19.0
        plan = plan_transfers(squad, proj, prices, clubs, positions, [101, 102],
                              max_transfers=2, hit_threshold=HIT_COST)
        self.assertEqual(len(plan.moves), 2)
        self.assertEqual(plan.hits, HIT_COST)

    def test_a_marginal_gain_does_not_burn_a_free_transfer_when_gated(self):
        squad, proj, prices, clubs, positions = self._world()
        # The best available move is worth +0.5: a real improvement, but a small one.
        # Forwards 13/14/15 project 13/14/15, so the upgrade is over the weakest.
        proj[101] = 13.5
        ungated = plan_transfers(squad, proj, prices, clubs, positions, [101],
                                 free_threshold=0.0)
        self.assertEqual(len(ungated.moves), 1, "there IS an improving transfer here")
        self.assertAlmostEqual(ungated.gain, 0.5)

        # Gate it above that and the same move must be declined — proving it is the
        # threshold biting, not an absence of candidates.
        gated = plan_transfers(squad, proj, prices, clubs, positions, [101],
                               free_threshold=1.0)
        self.assertEqual(gated.moves, [], "a free transfer is an option worth keeping")

    def test_planning_does_not_mutate_the_squad(self):
        squad, proj, prices, clubs, positions = self._world()
        before = list(squad.players), squad.bank, squad.free_transfers
        plan_transfers(squad, proj, prices, clubs, positions, [101])
        self.assertEqual((list(squad.players), squad.bank, squad.free_transfers), before)


class TestScoring(unittest.TestCase):
    def test_the_xi_is_picked_on_projections_and_scored_on_actuals(self):
        """Hindsight check: a player who blanked but was projected well must still be
        in the XI, and a bench player who hauled must NOT be counted."""
        positions, _, prices, proj = _squad_world()
        actuals = {pid: 0.0 for pid in range(1, 16)}
        actuals[1] = 100.0                      # worst projected, best actual: benched
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)}, season="2025-26")
        result = score_gameweek(squad, proj, actuals, positions, prices, gameweek=7)
        self.assertNotIn(1, result.xi)
        self.assertEqual(result.points, 0.0, "the bench scores nothing in FPL")

    def test_the_captain_is_doubled_and_hits_are_subtracted(self):
        positions, _, prices, proj = _squad_world()
        actuals = {pid: 2.0 for pid in range(1, 16)}
        squad = Squad(list(range(1, 16)), {i: 50 for i in range(1, 16)}, season="2025-26")
        from prediction_engine.fpl.manager import TransferPlan
        plan = TransferPlan(moves=[{}], hits=HIT_COST, gain=0.0)
        result = score_gameweek(squad, proj, actuals, positions, prices, 7, plan)
        self.assertEqual(result.points, 11 * 2.0 + 2.0)     # eleven starters + captain
        self.assertEqual(result.net, result.points - HIT_COST)


class TestOpeningSquad(unittest.TestCase):
    def test_it_buys_fifteen_legal_players_inside_the_budget(self):
        rows, pid = [], 0
        for club in range(8):
            for position in (GKP, GKP, DEF, DEF, MID, MID, FWD, FWD):
                pid += 1
                rows.append({"player_id": pid, "position": position,
                             "club": "C%d" % club, "price": 4.0 + (pid % 9) * 0.3,
                             "value": round((pid * 31 % 97) / 11.0, 4)})
        frame = pd.DataFrame(rows)
        squad = opening_squad(frame, "value", budget=100.0, season="2025-26")
        self.assertIsNotNone(squad)
        self.assertEqual(len(squad.players), 15)
        self.assertGreaterEqual(squad.bank, 0, "a squad must be affordable")

        positions = dict(zip(frame["player_id"], frame["position"]))
        counts = {}
        for p in squad.players:
            counts[positions[p]] = counts.get(positions[p], 0) + 1
        self.assertEqual(counts, {GKP: 2, DEF: 5, MID: 5, FWD: 3})

        clubs = dict(zip(frame["player_id"], frame["club"]))
        club_counts = {}
        for p in squad.players:
            club_counts[clubs[p]] = club_counts.get(clubs[p], 0) + 1
        self.assertLessEqual(max(club_counts.values()), 3)


if __name__ == "__main__":
    unittest.main()
