"""Is the XI optimizer actually optimal, or just plausible?

Branch and bound is only worth anything if its pruning is sound. A bound that is
slightly too aggressive silently returns a good-but-not-best team, and every
downstream benchmark inherits a quiet bias toward whichever model happens to suit
the search order. So the optimizer is checked against EXHAUSTIVE enumeration on
random instances small enough to brute-force, with and without a budget.

If these tests pass, "our team-picker was worse than theirs" is no longer an
available explanation for a negative benchmark result.
"""

import itertools
import random
import unittest

import pandas as pd

from prediction_engine.fpl.optimizer import (
    FORMATIONS,
    MAX_PER_CLUB,
    SQUAD_SIZE,
    drop_dominated,
    select_xi,
    xi_actual_points,
)

POSITION_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}

# 15 players -> C(15,11) = 1,365 candidate squads per instance. Small enough that
# exhaustive enumeration is genuinely exhaustive, large enough that every
# constraint (formation, club cap, budget) actually binds.
SMALL_INSTANCE = {1: 2, 2: 5, 3: 5, 4: 3}


def _brute_force(rows, budget=None, max_per_club=MAX_PER_CLUB):
    """The definition of the right answer: try literally every legal XI.

    Deliberately dumb, and written on plain lists rather than the optimizer's data
    structures - a reference implementation that shared machinery with the thing it
    checks would be able to share its bugs too.
    """
    positions = [row["position"] for row in rows]
    clubs = [row["club"] for row in rows]
    prices = [row["price"] for row in rows]
    points = [row["points"] for row in rows]

    best = None
    for combination in itertools.combinations(range(len(rows)), SQUAD_SIZE):
        counts = [0] * 5
        for i in combination:
            counts[positions[i]] += 1
        if any(not (low <= counts[position] <= high)
               for position, (low, high) in POSITION_BOUNDS.items()):
            continue

        club_counts, legal = {}, True
        for i in combination:
            club = clubs[i]
            club_counts[club] = club_counts.get(club, 0) + 1
            if club_counts[club] > max_per_club:
                legal = False
                break
        if not legal:
            continue

        if budget is not None and sum(prices[i] for i in combination) > budget + 1e-9:
            continue

        value = sum(points[i] for i in combination)
        if best is None or value > best:
            best = value
    return best


def _random_rows(rng, per_position=None, clubs=4):
    rows = []
    for position, count in (per_position or SMALL_INSTANCE).items():
        for _ in range(count):
            rows.append({
                "position": position,
                "club": "club_%d" % rng.randrange(clubs),
                "price": round(rng.uniform(4.0, 13.0), 1),
                "points": round(rng.uniform(0.0, 12.0), 2),
            })
    return rows


class TestOptimalityAgainstBruteForce(unittest.TestCase):
    """The load-bearing tests: branch and bound must never prune the true optimum."""

    def _compare(self, rng, budget=None, max_per_club=MAX_PER_CLUB, clubs=4):
        rows = _random_rows(rng, clubs=clubs)
        expected = _brute_force(rows, budget=budget, max_per_club=max_per_club)
        selection = select_xi(pd.DataFrame(rows), "points",
                              budget=budget, max_per_club=max_per_club)

        if expected is None:
            self.assertIsNone(selection, "no legal XI exists, but one was returned")
            return
        self.assertIsNotNone(selection, "a legal XI exists, but none was returned")
        self.assertAlmostEqual(selection.projected, expected, places=6)

    def test_matches_brute_force_without_budget(self):
        rng = random.Random(20260710)
        for _ in range(150):
            self._compare(rng)

    def test_matches_brute_force_with_a_generous_budget(self):
        rng = random.Random(11)
        for _ in range(150):
            self._compare(rng, budget=95.0)

    def test_matches_brute_force_with_a_binding_budget(self):
        # Tight enough that the cost bound does real work - and could be wrong.
        rng = random.Random(22)
        for _ in range(150):
            self._compare(rng, budget=85.0)

    def test_matches_brute_force_with_a_brutal_budget(self):
        # Near-infeasible: whole formations must be rejected, and sometimes all of them.
        rng = random.Random(33)
        for _ in range(150):
            self._compare(rng, budget=76.0)

    def test_matches_brute_force_with_a_binding_club_cap(self):
        # 4 clubs x 3 = 12 slots for 11 players: the cap is nearly saturated.
        rng = random.Random(44)
        for _ in range(150):
            self._compare(rng, clubs=4, max_per_club=3)

    def test_matches_brute_force_when_budget_and_club_cap_both_bind(self):
        rng = random.Random(55)
        for _ in range(150):
            self._compare(rng, budget=82.0, clubs=4, max_per_club=3)

    def test_matches_brute_force_on_a_larger_instance(self):
        # C(19,11) = 75,582 squads. Fewer repeats, more room for the bound to slip.
        rng = random.Random(66)
        for _ in range(6):
            rows = _random_rows(rng, {1: 3, 2: 6, 3: 6, 4: 4}, clubs=5)
            expected = _brute_force(rows, budget=88.0)
            selection = select_xi(pd.DataFrame(rows), "points", budget=88.0)
            if expected is None:
                self.assertIsNone(selection)
            else:
                self.assertAlmostEqual(selection.projected, expected, places=6)

    def test_matches_brute_force_when_many_players_tie(self):
        # Ties stress the strict `<=` in the value bound.
        rng = random.Random(77)
        for _ in range(60):
            rows = _random_rows(rng)
            for row in rows:
                row["points"] = float(rng.randrange(3))     # heavy ties
            expected = _brute_force(rows)
            selection = select_xi(pd.DataFrame(rows), "points")
            if expected is None:
                self.assertIsNone(selection)
            else:
                self.assertAlmostEqual(selection.projected, expected, places=6)


# The instance that exposed the float bug, found by the brute-force comparison.
# Its optimal XI (a 5-3-2) costs exactly 82.0 and scores 76.79.
_EXACT_BUDGET_INSTANCE = [
    (1, "club_3", 11.7, 6.06), (1, "club_1", 12.6, 6.79),
    (2, "club_2", 4.8, 11.85), (2, "club_2", 4.2, 0.68), (2, "club_0", 9.3, 4.78),
    (2, "club_0", 9.4, 5.12), (2, "club_1", 4.1, 11.72),
    (3, "club_3", 11.7, 4.31), (3, "club_2", 9.9, 6.42), (3, "club_3", 4.5, 8.00),
    (3, "club_3", 11.9, 1.25), (3, "club_1", 7.5, 11.05),
    (4, "club_3", 8.4, 2.92), (4, "club_2", 10.8, 2.21), (4, "club_2", 5.5, 9.57),
]


class TestBudgetArithmeticIsExact(unittest.TestCase):
    """An optimal FPL team spends its whole budget, so the XI costing EXACTLY the
    limit is the one that matters most. Accumulated in the optimizer's own order its
    price sums to 82.00000000000001 and it is silently discarded - in every
    gameweek, for every model, always throwing away the answer. Integer tenths make
    that impossible. Found by the brute-force comparison above, not by inspection.
    """

    def test_the_float_trap_is_real_and_order_dependent(self):
        # Why this survived a reading of the code: sum() in list order is exact.
        prices = [12.6, 4.8, 4.2, 9.3, 9.4, 4.1, 11.7, 4.5, 7.5, 8.4, 5.5]
        self.assertEqual(sum(prices), 82.0)

        # Accumulated the way the solver does it (GKP, then DEF/MID/FWD by points):
        running = 0.0
        for price in [12.6, 4.8, 4.1, 9.4, 9.3, 4.2, 7.5, 4.5, 11.7, 5.5, 8.4]:
            running += price
        self.assertGreater(running, 82.0)
        self.assertEqual(sum(int(round(p * 10)) for p in prices), 820)   # exact in tenths

    def _instance(self):
        return pd.DataFrame(
            [{"position": position, "club": club, "price": price, "points": points}
             for position, club, price, points in _EXACT_BUDGET_INSTANCE])

    def test_finds_the_exactly_affordable_optimum(self):
        players = self._instance()
        expected = _brute_force(players.to_dict("records"), budget=82.0, max_per_club=3)
        selection = select_xi(players, "points", budget=82.0, max_per_club=3)

        self.assertIsNotNone(selection, "the exactly-affordable optimum was rejected")
        self.assertAlmostEqual(selection.projected, expected, places=6)
        self.assertAlmostEqual(selection.projected, 76.79, places=2)
        self.assertAlmostEqual(selection.cost, 82.0, places=6)
        self.assertEqual(selection.formation, (5, 3, 2))

    def test_one_tenth_under_the_budget_excludes_that_optimum(self):
        players = self._instance()
        selection = select_xi(players, "points", budget=81.9, max_per_club=3)
        if selection is not None:
            self.assertLess(selection.projected, 76.79)
            self.assertLessEqual(selection.cost, 81.9 + 1e-9)


class TestDominanceReductionIsSafe(unittest.TestCase):
    """`drop_dominated` deletes players before the search ever sees them. If its
    reasoning is wrong the optimizer returns a confidently wrong team, so it is
    checked against brute force over the FULL, unreduced pool.
    """

    def test_identical_players_leave_exactly_max_per_club_survivors(self):
        # A club may field 3, so 3 of 6 identical players must survive - not 1, not 0.
        rows = [{"position": 2, "club": "a", "price": 5.0, "points": 4.0} for _ in range(6)]
        kept = drop_dominated(pd.DataFrame(rows), "points", max_per_club=3)
        self.assertEqual(len(kept), 3)

    def test_a_worse_and_dearer_player_is_dropped(self):
        rows = [
            {"position": 2, "club": "a", "price": 4.0, "points": 9.0},
            {"position": 2, "club": "a", "price": 4.5, "points": 8.0},
            {"position": 2, "club": "a", "price": 5.0, "points": 7.0},
            {"position": 2, "club": "a", "price": 9.9, "points": 1.0},   # dominated by 3
        ]
        kept = drop_dominated(pd.DataFrame(rows), "points", max_per_club=3)
        self.assertEqual(len(kept), 3)
        self.assertNotIn(3, list(kept.index))

    def test_a_cheaper_player_is_never_dropped_even_if_worse(self):
        # He may be exactly what a tight budget needs.
        rows = [
            {"position": 2, "club": "a", "price": 9.0, "points": 9.0},
            {"position": 2, "club": "a", "price": 9.0, "points": 8.0},
            {"position": 2, "club": "a", "price": 9.0, "points": 7.0},
            {"position": 2, "club": "a", "price": 4.0, "points": 1.0},   # cheapest: keep
        ]
        kept = drop_dominated(pd.DataFrame(rows), "points", max_per_club=3)
        self.assertIn(3, list(kept.index))

    def test_players_from_a_different_club_never_dominate(self):
        rows = [
            {"position": 2, "club": "a", "price": 4.0, "points": 9.0},
            {"position": 2, "club": "a", "price": 4.0, "points": 9.0},
            {"position": 2, "club": "a", "price": 4.0, "points": 9.0},
            {"position": 2, "club": "b", "price": 9.9, "points": 1.0},   # other club: keep
        ]
        kept = drop_dominated(pd.DataFrame(rows), "points", max_per_club=3)
        self.assertIn(3, list(kept.index))

    def _concentrated_rows(self, rng):
        """Five defenders per club, and better players are CHEAPER.

        With independent random prices a player almost never accumulates three
        dominators, so the reduction never fires and a test built on it proves
        nothing. Anti-correlating price with points (plus jitter, so it is not a
        perfect order) makes domination common - the case the reduction must get
        right. 18 players: C(18,11) = 31,824 squads, still exhaustively checkable.
        """
        rows = [{"position": 1, "club": "a"}, {"position": 1, "club": "b"}]
        for club in ("a", "b"):
            rows.extend({"position": 2, "club": club} for _ in range(5))
        for club in ("c", "d"):
            rows.extend({"position": 3, "club": club} for _ in range(2))
        rows.extend({"position": 4, "club": club} for club in ("c", "d"))

        for row in rows:
            points = round(rng.uniform(0.0, 12.0), 2)
            row["points"] = points
            row["price"] = round(4.0 + (12.0 - points) * 0.4 + rng.uniform(-0.5, 0.5), 1)
        return rows

    def test_reduction_actually_fires_on_concentrated_pools(self):
        fired = 0
        rng = random.Random(3)
        for _ in range(20):
            players = pd.DataFrame(self._concentrated_rows(rng))
            if len(drop_dominated(players, "points")) < len(players):
                fired += 1
        self.assertGreater(fired, 10, "the reduction rarely triggered; the safety "
                                      "test below would prove nothing")

    def test_optimum_is_unchanged_by_the_reduction(self):
        rng = random.Random(101)
        for _ in range(15):
            rows = self._concentrated_rows(rng)
            for budget in (None, 70.0, 62.0):
                expected = _brute_force(rows, budget=budget)      # full, unreduced pool
                selection = select_xi(pd.DataFrame(rows), "points", budget=budget)
                if expected is None:
                    self.assertIsNone(selection)
                else:
                    self.assertAlmostEqual(selection.projected, expected, places=6)

    def test_reduction_is_idempotent(self):
        players = pd.DataFrame(self._concentrated_rows(random.Random(9)))
        once = drop_dominated(players, "points")
        twice = drop_dominated(once, "points")
        self.assertEqual(list(once.index), list(twice.index))


class TestConstraintsAreRespected(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.players = pd.DataFrame(_random_rows(rng, {1: 6, 2: 12, 3: 12, 4: 8}, clubs=8))

    def test_formation_is_legal(self):
        squad = self.players.loc[select_xi(self.players, "points").labels]
        counts = squad["position"].value_counts()
        self.assertEqual(len(squad), SQUAD_SIZE)
        for position, (low, high) in POSITION_BOUNDS.items():
            self.assertTrue(low <= counts.get(position, 0) <= high)

    def test_never_more_than_three_from_one_club(self):
        squad = self.players.loc[select_xi(self.players, "points").labels]
        self.assertLessEqual(squad["club"].value_counts().max(), MAX_PER_CLUB)

    def test_budget_is_never_exceeded(self):
        selection = select_xi(self.players, "points", budget=75.0)
        self.assertIsNotNone(selection)
        self.assertLessEqual(selection.cost, 75.0 + 1e-9)

    def test_tighter_budget_never_scores_more(self):
        loose = select_xi(self.players, "points", budget=90.0)
        tight = select_xi(self.players, "points", budget=70.0)
        self.assertLessEqual(tight.projected, loose.projected + 1e-9)

    def test_returns_none_when_no_legal_xi_exists(self):
        no_keeper = self.players[self.players["position"] != 1]
        self.assertIsNone(select_xi(no_keeper, "points"))

    def test_returns_none_when_budget_is_impossible(self):
        self.assertIsNone(select_xi(self.players, "points", budget=1.0))

    def test_all_formations_are_legal_and_complete(self):
        self.assertEqual(len(FORMATIONS), 8)
        for defenders, midfielders, forwards in FORMATIONS:
            self.assertEqual(1 + defenders + midfielders + forwards, SQUAD_SIZE)


class TestCaptainAndScoring(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame([
            {"position": 1, "club": "a", "price": 5.0, "points": 3.0, "actual": 2},
            {"position": 2, "club": "a", "price": 5.0, "points": 4.0, "actual": 5},
            {"position": 2, "club": "b", "price": 5.0, "points": 4.0, "actual": 5},
            {"position": 2, "club": "c", "price": 5.0, "points": 4.0, "actual": 5},
            {"position": 3, "club": "b", "price": 5.0, "points": 9.0, "actual": 11},
            {"position": 3, "club": "c", "price": 5.0, "points": 4.0, "actual": 1},
            {"position": 3, "club": "d", "price": 5.0, "points": 4.0, "actual": 1},
            {"position": 3, "club": "d", "price": 5.0, "points": 4.0, "actual": 1},
            {"position": 3, "club": "e", "price": 5.0, "points": 4.0, "actual": 1},
            {"position": 4, "club": "e", "price": 5.0, "points": 4.0, "actual": 3},
            {"position": 4, "club": "f", "price": 5.0, "points": 4.0, "actual": 3},
            {"position": 4, "club": "f", "price": 5.0, "points": 1.0, "actual": 0},
        ])

    def test_captain_is_the_highest_projected_player_in_the_xi(self):
        selection = select_xi(self.players, "points")
        self.assertIn(selection.captain, selection.labels)
        self.assertEqual(self.players.loc[selection.captain, "points"], 9.0)

    def test_captain_score_counts_twice(self):
        selection = select_xi(self.players, "points")
        plain = xi_actual_points(self.players, selection, with_captain=False)
        doubled = xi_actual_points(self.players, selection, with_captain=True)
        self.assertEqual(doubled - plain, 11)   # the captain's own actual score

    def test_cost_matches_the_selected_squad(self):
        selection = select_xi(self.players, "points")
        self.assertAlmostEqual(selection.cost, 55.0)


class TestDegenerateValueColumns(unittest.TestCase):
    """The incumbent and the bound add the same points in different orders, so on a
    tied XI they disagree by ~1e-15. Without slack the prune never fires and the
    search enumerates everything: a constant column (the `global_mean` baseline)
    hung forever. These pin the fix - they must finish, not merely pass."""

    def _pool(self, value):
        rows = []
        for position, count in ((1, 6), (2, 20), (3, 20), (4, 12)):
            for i in range(count):
                rows.append({"position": position, "club": "club_%d" % (i % 8),
                             "price": 4.0 + (i % 9) * 0.5, "points": value})
        return pd.DataFrame(rows)

    def test_constant_value_column_terminates_unbudgeted(self):
        selection = select_xi(self._pool(1.55), "points")
        self.assertIsNotNone(selection)
        self.assertEqual(len(selection.labels), SQUAD_SIZE)
        self.assertAlmostEqual(selection.projected, 11 * 1.55, places=6)

    def test_constant_value_column_terminates_with_a_budget(self):
        selection = select_xi(self._pool(1.55), "points", budget=60.0)
        self.assertIsNotNone(selection)
        self.assertLessEqual(selection.cost, 60.0 + 1e-9)
        self.assertAlmostEqual(selection.projected, 11 * 1.55, places=6)

    def test_all_zero_value_column_terminates(self):
        selection = select_xi(self._pool(0.0), "points", budget=70.0)
        self.assertIsNotNone(selection)
        self.assertEqual(selection.projected, 0.0)

    def test_near_ties_still_find_the_true_optimum(self):
        # One player is better by far more than BOUND_TOLERANCE: he must be picked.
        players = self._pool(1.0)
        players.loc[players.index[-1], "points"] = 5.0     # a forward
        selection = select_xi(players, "points")
        self.assertIn(players.index[-1], selection.labels)
        self.assertAlmostEqual(selection.projected, 10 * 1.0 + 5.0, places=6)


class TestInputValidation(unittest.TestCase):
    def test_missing_column_raises(self):
        frame = pd.DataFrame({"position": [1], "club": ["a"], "price": [4.0]})
        with self.assertRaises(ValueError):
            select_xi(frame, "points")

    def test_too_few_players_returns_none(self):
        frame = pd.DataFrame([{"position": 1, "club": "a", "price": 4.0, "points": 1.0}])
        self.assertIsNone(select_xi(frame, "points"))


if __name__ == "__main__":
    unittest.main()
