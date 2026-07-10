"""Is `select_squad` really solving the game FPL actually plays?

The rules it must obey, all at once:

    squad         exactly 2 GKP, 5 DEF, 5 MID, 3 FWD  (fifteen players)
    budget        the whole squad costs <= 100.0m - the bench is NOT free
    club cap      at most 3 players from any one club, across ALL FIFTEEN
    formation     the eleven who start: 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD
    scoring       only the starting eleven score; the bench exists to be cheap

The interesting consequence, and the reason a flat "XI budget" is a fiction: the
bench is *forced* by the formation, because the squad quota is fixed. A 5-4-1
must bench two forwards (dear); a 3-4-3 benches two defenders (cheap).

Checked against exhaustive enumeration: every legal XI, and for each of them every
legal bench. If branch and bound ever disagrees, these fail.
"""

import itertools
import random
import unittest

import pandas as pd

from prediction_engine.fpl.optimizer import (
    BENCH_SIZE,
    FULL_SQUAD_SIZE,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    select_squad,
    xi_actual_points,
)

POSITION_BOUNDS = {1: (1, 1), 2: (3, 5), 3: (2, 5), 4: (1, 3)}


def _brute_force_squad(rows, budget=100.0, max_per_club=MAX_PER_CLUB):
    """Every legal XI, every legal bench. The definition of the right answer."""
    indices = range(len(rows))
    best = None

    for xi in itertools.combinations(indices, SQUAD_SIZE):
        counts = [0] * 5
        for i in xi:
            counts[rows[i]["position"]] += 1
        if any(not (low <= counts[p] <= high) for p, (low, high) in POSITION_BOUNDS.items()):
            continue

        clubs = {}
        for i in xi:
            clubs[rows[i]["club"]] = clubs.get(rows[i]["club"], 0) + 1
        if max(clubs.values()) > max_per_club:
            continue

        value = sum(rows[i]["points"] for i in xi)
        if best is not None and value <= best:
            continue                      # cannot improve; skip the bench search

        xi_cost = sum(rows[i]["price"] for i in xi)
        bench_needs = {p: SQUAD_QUOTA[p] - counts[p] for p in SQUAD_QUOTA}
        remaining = [i for i in indices if i not in xi]

        for bench in itertools.combinations(remaining, BENCH_SIZE):
            bench_counts = [0] * 5
            for i in bench:
                bench_counts[rows[i]["position"]] += 1
            if any(bench_counts[p] != bench_needs[p] for p in SQUAD_QUOTA):
                continue
            squad_clubs = dict(clubs)
            for i in bench:
                squad_clubs[rows[i]["club"]] = squad_clubs.get(rows[i]["club"], 0) + 1
            if max(squad_clubs.values()) > max_per_club:
                continue
            if xi_cost + sum(rows[i]["price"] for i in bench) > budget + 1e-9:
                continue
            best = value                  # a legal, affordable squad exists for this XI
            break
    return best


def _random_rows(rng, clubs=6, per_position=None):
    """19 players: C(19,11) = 75,582 starting elevens, each with every legal bench
    enumerated behind it. Small enough to be genuinely exhaustive, large enough that
    the squad quota (needing 15 of the 19), the budget and the club cap all bind."""
    per_position = per_position or {1: 3, 2: 6, 3: 6, 4: 4}
    rows = []
    for position, count in per_position.items():
        for _ in range(count):
            rows.append({
                "position": position,
                "club": "club_%d" % rng.randrange(clubs),
                "price": round(rng.uniform(4.0, 12.0), 1),
                "points": round(rng.uniform(0.0, 10.0), 2),
            })
    return rows


class TestSquadMatchesBruteForce(unittest.TestCase):
    def _compare(self, rng, budget, clubs=6):
        rows = _random_rows(rng, clubs=clubs)
        expected = _brute_force_squad(rows, budget=budget)
        selection = select_squad(pd.DataFrame(rows), "points", squad_budget=budget)

        if expected is None:
            self.assertIsNone(selection, "no legal squad exists, but one was returned")
            return
        self.assertIsNotNone(selection, "a legal squad exists, but none was returned")
        self.assertAlmostEqual(selection.projected, expected, places=6)

    def test_matches_brute_force_with_a_generous_budget(self):
        rng = random.Random(2026)
        for _ in range(8):
            self._compare(rng, budget=140.0)

    def test_matches_brute_force_with_a_binding_budget(self):
        rng = random.Random(7)
        for _ in range(8):
            self._compare(rng, budget=115.0)

    def test_matches_brute_force_with_a_brutal_budget(self):
        # Near-infeasible: most starting elevens have no affordable bench behind them.
        rng = random.Random(99)
        for _ in range(8):
            self._compare(rng, budget=100.0)

    def test_matches_brute_force_with_a_binding_club_cap(self):
        # 5 clubs x 3 = 15 slots for a squad of 15: the cap is fully saturated.
        rng = random.Random(31)
        for _ in range(8):
            self._compare(rng, budget=120.0, clubs=5)

    def test_matches_brute_force_when_the_bench_forces_the_formation(self):
        # Forwards dear, defenders cheap: benching two forwards (5-4-1) is punished,
        # benching two defenders (3-4-3) is not. The optimum must feel that.
        rng = random.Random(64)
        for _ in range(8):
            rows = _random_rows(rng)
            for row in rows:
                row["price"] = 4.0 + (6.0 if row["position"] == 4 else 0.5) \
                    + round(rng.uniform(0.0, 2.0), 1)
            expected = _brute_force_squad(rows, budget=100.0)
            selection = select_squad(pd.DataFrame(rows), "points", squad_budget=100.0)
            if expected is None:
                self.assertIsNone(selection)
            else:
                self.assertAlmostEqual(selection.projected, expected, places=6)


class TestSquadRules(unittest.TestCase):
    def setUp(self):
        rng = random.Random(11)
        self.players = pd.DataFrame(_random_rows(rng, clubs=8,
                                                 per_position={1: 8, 2: 20, 3: 20, 4: 12}))
        self.selection = select_squad(self.players, "points", squad_budget=100.0)

    def test_squad_is_fifteen_with_the_right_quota(self):
        squad = self.players.loc[self.selection.xi + self.selection.bench]
        self.assertEqual(len(squad), FULL_SQUAD_SIZE)
        counts = squad["position"].value_counts()
        for position, quota in SQUAD_QUOTA.items():
            self.assertEqual(counts.get(position, 0), quota)

    def test_starting_eleven_is_a_legal_formation(self):
        xi = self.players.loc[self.selection.xi]
        self.assertEqual(len(xi), SQUAD_SIZE)
        counts = xi["position"].value_counts()
        for position, (low, high) in POSITION_BOUNDS.items():
            self.assertTrue(low <= counts.get(position, 0) <= high)

    def test_club_cap_applies_across_all_fifteen(self):
        squad = self.players.loc[self.selection.xi + self.selection.bench]
        self.assertLessEqual(squad["club"].value_counts().max(), MAX_PER_CLUB)

    def test_bench_is_never_free(self):
        squad = self.players.loc[self.selection.xi + self.selection.bench]
        self.assertAlmostEqual(self.selection.cost, squad["price"].sum(), places=6)
        self.assertLessEqual(self.selection.cost, 100.0 + 1e-9)
        bench_cost = self.players.loc[self.selection.bench, "price"].sum()
        self.assertGreater(bench_cost, 0.0)

    def test_xi_and_bench_are_disjoint(self):
        self.assertFalse(set(self.selection.xi) & set(self.selection.bench))

    def test_bench_composition_is_forced_by_the_formation(self):
        defenders, midfielders, forwards = self.selection.formation
        bench = self.players.loc[self.selection.bench]["position"].value_counts()
        self.assertEqual(bench.get(1, 0), 1)                      # always the backup GK
        self.assertEqual(bench.get(2, 0), SQUAD_QUOTA[2] - defenders)
        self.assertEqual(bench.get(3, 0), SQUAD_QUOTA[3] - midfielders)
        self.assertEqual(bench.get(4, 0), SQUAD_QUOTA[4] - forwards)

    def test_only_the_starting_eleven_scores(self):
        players = self.players.copy()
        players["actual"] = 0
        players.loc[self.selection.bench, "actual"] = 100      # bench hauls, worth nothing
        self.assertEqual(xi_actual_points(players, self.selection), 0.0)

    def test_captain_starts_and_doubles(self):
        players = self.players.copy()
        players["actual"] = 0
        players.loc[self.selection.captain, "actual"] = 7
        self.assertIn(self.selection.captain, self.selection.xi)
        plain = xi_actual_points(players, self.selection, with_captain=False)
        doubled = xi_actual_points(players, self.selection, with_captain=True)
        self.assertEqual(doubled - plain, 7)

    def test_tighter_budget_never_scores_more(self):
        loose = select_squad(self.players, "points", squad_budget=110.0)
        tight = select_squad(self.players, "points", squad_budget=95.0)
        self.assertLessEqual(tight.projected, loose.projected + 1e-9)


class TestSquadInfeasibility(unittest.TestCase):
    def test_returns_none_without_two_goalkeepers(self):
        rng = random.Random(5)
        rows = _random_rows(rng, per_position={1: 1, 2: 7, 3: 7, 4: 5})
        self.assertIsNone(select_squad(pd.DataFrame(rows), "points"))

    def test_returns_none_when_the_squad_is_unaffordable(self):
        rng = random.Random(5)
        rows = _random_rows(rng)
        self.assertIsNone(select_squad(pd.DataFrame(rows), "points", squad_budget=10.0))

    def test_missing_column_raises(self):
        frame = pd.DataFrame({"position": [1], "club": ["a"], "price": [4.0]})
        with self.assertRaises(ValueError):
            select_squad(frame, "points")


if __name__ == "__main__":
    unittest.main()
