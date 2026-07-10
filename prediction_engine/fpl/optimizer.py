"""Pick the best XI a manager could ACTUALLY field, exactly.

The backtest used to rank players by projected points and take the top eleven.
That is not a team - it fielded 5.9 goalkeepers, and FPL allows one. Replacing it
with a greedy legal fill was better but still a heuristic, which left "our
team-picker was simply worse than theirs" alive as an explanation for any negative
result. This module removes that excuse: given each player's projected points it
returns the *provably* highest-scoring legal XI, so any remaining difference
between two models is a difference in their PROJECTIONS, not in how the team was
assembled.

Constraints, all of them real FPL rules:

    formation     exactly 1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD   (8 legal shapes)
    club cap      at most 3 players from any one club
    budget        optional; total price must not exceed it

Adding a budget makes this a multi-dimensional knapsack, which is NP-hard, so
there is no closed form. It is solved by branch and bound over the eight
formations, and the search is exact - not "good enough":

    value bound   within a position players are visited in descending projected
                  points, so the best any completion can reach is the current
                  total plus the next k players here plus the unconstrained best
                  of every later position. Once that ceiling falls to the
                  incumbent, no later index can beat it either, so we stop.
    cost bound    the cheapest possible completion (k cheapest remaining here,
                  plus the cheapest legal fill of every later position) is
                  precomputed. If even that busts the budget, prune. It only
                  grows as we advance, so we stop rather than skip.

`select_xi` is verified against exhaustive enumeration on thousands of random
instances in `prediction_engine/tests/test_fpl_optimizer.py`. Where brute force is
tractable, they agree exactly.

The captain (FPL doubles one player's score) is chosen AFTER the XI, as the
highest-projected player in it. That is the order a manager actually decides in,
and it is not jointly optimal in principle - an XI could in theory be rebuilt
around a captain. In practice the highest-projected player is in the optimal XI
essentially always, so the two coincide. Stated rather than hidden.
"""

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID

logger = logging.getLogger(__name__)

SQUAD_SIZE = 11
MAX_PER_CLUB = 3
POSITIONS: Tuple[int, ...] = (GKP, DEF, MID, FWD)

# The REAL game. You buy fifteen players for 100.0m - never eleven - and start
# eleven of them. The other four are bought from the same pool, out of the same
# money, and count against the same 3-per-club limit. They score nothing.
#
# So there is no such thing as an "XI budget": it is whatever is left after the
# bench, and the bench is FORCED by the formation, because the squad quota is
# fixed. Start five defenders and your bench must carry two forwards, who are
# dear. Start three and it carries two defenders, who are cheap. Measured on real
# prices, that swing is 83.1m (5-4-1) to 84.3m (3-4-3) - so any "XI budget" above
# ~84.3m describes a squad that cannot be bought.
SQUAD_QUOTA: Dict[int, int] = {GKP: 2, DEF: 5, MID: 5, FWD: 3}
FULL_SQUAD_SIZE = sum(SQUAD_QUOTA.values())          # 15
BENCH_SIZE = FULL_SQUAD_SIZE - SQUAD_SIZE            # 4
TOTAL_SQUAD_BUDGET = 100.0

# Every legal FPL formation, as (defenders, midfielders, forwards). One keeper always.
FORMATIONS: Tuple[Tuple[int, int, int], ...] = tuple(
    (defenders, midfielders, forwards)
    for defenders in range(3, 6)
    for midfielders in range(2, 6)
    for forwards in range(1, 4)
    if defenders + midfielders + forwards == SQUAD_SIZE - 1
)

MAX_SLOTS_PER_POSITION = 5   # the largest count any single position can require

# Prices are exact multiples of £0.1m, so all budget arithmetic is done in integer
# tenths. This is not fussiness. In floating point, 12.6+4.8+4.2+... = 82.000000000000014,
# and a team costing exactly the budget is rejected as too expensive. An optimal FPL
# team *always* spends its entire budget, so a float comparison throws away precisely
# the answer, in every gameweek, for every model. Integers make it impossible.
TENTHS_PER_MILLION = 10
UNAFFORDABLE = 1 << 40       # an integer stand-in for infinity
IMPOSSIBLE = -1e18           # a float stand-in for "no such selection exists"

# The value bound and the incumbent sum the same points in different orders, so
# they can disagree by ~1e-15 on an XI that is genuinely tied. Without slack the
# prune never fires and the search enumerates the whole space - a constant-valued
# column (the `global_mean` baseline) hangs outright, and finely-tied projections
# crawl. Pruning at `best + BOUND_TOLERANCE` can only discard teams within a
# billionth of a point of the optimum, which is far below the resolution of a
# metric denominated in whole FPL points.
BOUND_TOLERANCE = 1e-9

XiSelection = namedtuple("XiSelection", "labels projected cost formation captain")


def _to_tenths(price) -> int:
    return int(round(float(price) * TENTHS_PER_MILLION))


class _PositionPool(object):
    """One position's candidates, sorted by descending projected points.

    Descending order is what makes the value bound valid: the k best players
    reachable from index i are exactly i..i+k-1.
    """

    __slots__ = ("values", "prices", "clubs", "labels", "min_cost", "size", "affordable")

    def __init__(self, frame: pd.DataFrame, value_column: str):
        ordered = frame.sort_values(value_column, ascending=False)
        self.values = ordered[value_column].to_numpy(dtype=float)
        self.prices = [_to_tenths(price) for price in ordered["price"]]
        self.clubs = ordered["club"].to_numpy()
        self.labels = list(ordered.index)
        self.size = len(ordered)
        self.min_cost = _suffix_min_cost(self.prices)
        self.affordable = None      # set by `build_affordable_table` when budgeted

    def best_values(self, count: int) -> float:
        return float(self.values[:count].sum()) if self.size >= count else 0.0

    def cheapest_fill(self, count: int) -> int:
        return self.min_cost[count][0]

    def build_affordable_table(self, budget_tenths: int) -> None:
        self.affordable = _affordable_table(self.prices, self.values, budget_tenths)


def _affordable_table(prices: Sequence[int], values: np.ndarray, budget: int) -> np.ndarray:
    """`table[k][b]` = the most value obtainable from EXACTLY `k` of these players
    for at most `b` tenths of a million. Club caps are ignored, so it can only
    over-state - which is exactly what an upper bound must do.

    Why this exists: the plain value bound ("the k best players here, ignoring
    price") is nearly useless once a budget bites, because the k best are usually
    unaffordable together. Worse, it degrades precisely for a model whose
    projections are finely spaced and never tie - which is ours. Measured on a real
    gameweek, `ours` at £83.0m took 39 seconds while `player_ppg` took 0.035s,
    purely because integer-average ties let the old bound prune and our continuous
    values did not. A budget-aware bound removes that asymmetry, and with it the
    temptation to swap exactness for speed on the one model we most need to trust.

    Standard 0/1 knapsack with an exact-cardinality dimension: `k` descends so no
    player is used twice.
    """
    table = np.full((MAX_SLOTS_PER_POSITION + 1, budget + 1), IMPOSSIBLE, dtype=float)
    table[0, :] = 0.0
    for price, value in zip(prices, values):
        if price > budget:
            continue
        for count in range(MAX_SLOTS_PER_POSITION, 0, -1):
            np.maximum(table[count, price:], table[count - 1, :budget + 1 - price] + value,
                       out=table[count, price:])
    return table


def _max_plus_convolve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """`out[b] = max over b1 <= b of left[b1] + right[b - b1]`.

    Splits a budget optimally between two independent groups of players.

    Both inputs are non-decreasing in the budget (more money never hurts), so if
    `left[b1] == left[b1 - 1]` then spending `b1` is pointless: the same value is
    available for less, and the leftover can only help `right`. Only the budgets at
    which `left` actually improves need to be considered, which in practice is a
    small fraction of them.
    """
    budget = len(left) - 1
    out = np.full(budget + 1, IMPOSSIBLE, dtype=float)

    improves = np.empty(budget + 1, dtype=bool)
    improves[0] = True
    np.greater(left[1:], left[:-1], out=improves[1:])

    for spent in np.flatnonzero(improves):
        if left[spent] <= IMPOSSIBLE / 2:
            continue
        np.maximum(out[spent:], left[spent] + right[:budget + 1 - spent], out=out[spent:])
    return out


def _node_bounds(pools, counts, budget: int) -> List[np.ndarray]:
    """`out[p][k][b]` = the most value obtainable by taking `k` more players from
    position `p` AND filling every later position, with `b` tenths left to spend.

    This is the bound that makes the search tractable. The obvious version - "the
    best `k` here, plus the best fill of each later position" - lets *every* group
    spend the entire remaining budget, so it wildly over-states once money is tight
    and prunes almost nothing. Measured: `ours` at £80.0m took 115 seconds with
    that bound. Sharing the budget across groups by max-plus convolution makes the
    bound nearly exact (only the club cap is still relaxed), and the search
    collapses.

    Still an upper bound - it ignores the 3-per-club rule - so branch and bound
    remains exact. The brute-force suite is what actually checks that claim.
    """
    tail = np.zeros(budget + 1, dtype=float)          # nothing left to fill
    bounds: List[np.ndarray] = [None] * len(POSITIONS)
    for position_index in range(len(POSITIONS) - 1, -1, -1):
        pool = pools[position_index]
        needed = counts[position_index]
        table = np.full((needed + 1, budget + 1), IMPOSSIBLE, dtype=float)
        for count in range(needed + 1):
            # The pool's table may have been built for a larger budget (the squad's);
            # only the first `budget + 1` entries are reachable here.
            table[count] = _max_plus_convolve(pool.affordable[count][:budget + 1], tail)
        bounds[position_index] = table
        tail = table[needed]                          # this position, fully filled
    return bounds


def _suffix_min_cost(prices: Sequence[int]) -> List[List[int]]:
    """`out[k][i]` = cheapest total price (in tenths) of choosing `k` from `prices[i:]`.

    Non-decreasing in `i` (a shorter suffix can only be dearer), which is what lets
    the caller stop instead of merely skipping when the budget bound fails.
    """
    size = len(prices)
    out = [[0] * (size + 1) for _ in range(MAX_SLOTS_PER_POSITION + 1)]
    for k in range(1, MAX_SLOTS_PER_POSITION + 1):
        out[k][size] = UNAFFORDABLE

    smallest: List[int] = []            # the k cheapest prices seen in prices[i:]
    for i in range(size - 1, -1, -1):
        price = prices[i]
        position = len(smallest)
        while position > 0 and smallest[position - 1] > price:
            position -= 1
        smallest.insert(position, price)
        del smallest[MAX_SLOTS_PER_POSITION:]

        for k in range(1, MAX_SLOTS_PER_POSITION + 1):
            out[k][i] = sum(smallest[:k]) if len(smallest) >= k else UNAFFORDABLE
    return out


def drop_dominated(players: pd.DataFrame, value_column: str,
                   max_per_club: int = MAX_PER_CLUB) -> pd.DataFrame:
    """Remove players who can never appear in an optimal XI. Provably safe.

    Say `q` and `p` play the same position for the same club, and `q` projects at
    least as many points for no more money. Any XI containing `p` can swap in `q`:
    same club (the 3-per-club count is unchanged), same position (the formation is
    unchanged), no dearer (still affordable), no worse (value never drops). The swap
    is blocked only if `q` is already in the XI.

    At most `max_per_club` players come from any one club, so if `p` has
    `max_per_club` or more such dominators, at least one of them must be outside the
    XI and the swap is always available. Therefore an optimal XI exists without `p`,
    and dropping him cannot change the optimum.

    Ties are broken by (value desc, price asc, position in the frame) so that among
    identical players exactly `max_per_club` survive - never zero, which would be a
    subtle way to lose the answer.

    The same argument holds for a full 15-man squad, because FPL's 3-per-club limit
    applies to the whole squad, not just the eleven: if `p` is in the squad, at most
    `max_per_club - 1` other players from his club are, so with `max_per_club`
    dominators one is always free. Swapping keeps a starter no worse and makes a
    bench player no dearer.

    This is not a heuristic. The brute-force suites compare against exhaustive
    enumeration over the FULL pool, so an unsafe reduction would fail those tests.
    """
    keep = []
    for _, group in players.groupby(["position", "club"], sort=False):
        if len(group) <= max_per_club:
            keep.extend(group.index)
            continue

        ordered = group.sort_values([value_column, "price"], ascending=[False, True])
        prices_seen: List[float] = []       # prices of players already kept-or-dropped
        for label, price in zip(ordered.index, ordered["price"]):
            # Everyone earlier has value >= this player's; count those no dearer.
            dominators = sum(1 for seen in prices_seen if seen <= price)
            if dominators < max_per_club:
                keep.append(label)
            prices_seen.append(price)
    return players.loc[keep]


def _formation_needs(formation: Tuple[int, int, int]) -> Dict[int, int]:
    defenders, midfielders, forwards = formation
    return {GKP: 1, DEF: defenders, MID: midfielders, FWD: forwards}


def _greedy_xi(pools, counts, budget_tenths, max_per_club):
    """A decent legal XI, found fast, used only to SEED the search.

    Branch and bound prunes a branch when its ceiling cannot beat the best team
    found so far - so with no team found so far it prunes nothing, and the first
    dive is the value-greedy XI, which under a tight budget is unaffordable and
    backtracks endlessly. Handing the search a good team up front makes the bound
    bite from the first node.

    This is a lower bound on the optimum, never a substitute for it: the search
    still explores everything that could beat it. Measured on a real gameweek, this
    is what took `ours` at £80.0m from 115 seconds to well under one.

    Takes players in descending value, skipping any who would make the cheapest
    legal completion unaffordable, or whose club is full. Returns None if it paints
    itself into a corner - a missing seed costs speed, never correctness.
    """
    cost_after = [0] * (len(POSITIONS) + 1)
    for i in range(len(POSITIONS) - 1, -1, -1):
        cost_after[i] = cost_after[i + 1] + pools[i].cheapest_fill(counts[i])

    clubs: Dict[object, int] = {}
    chosen: List[Tuple[int, int]] = []
    value, cost = 0.0, 0

    for position_index in range(len(POSITIONS)):
        pool = pools[position_index]
        taken = 0
        for index in range(pool.size):
            if taken == counts[position_index]:
                break
            club = pool.clubs[index]
            if clubs.get(club, 0) >= max_per_club:
                continue
            price = pool.prices[index]
            if budget_tenths is not None:
                still_needed = counts[position_index] - taken - 1
                completion = (pool.min_cost[still_needed][index + 1]
                              + cost_after[position_index + 1])
                if cost + price + completion > budget_tenths:
                    continue
            clubs[club] = clubs.get(club, 0) + 1
            chosen.append((position_index, index))
            value += float(pool.values[index])
            cost += price
            taken += 1
        if taken < counts[position_index]:
            return None
    return value, [pools[p].labels[i] for p, i in chosen]


def _solve_formation(pools, needs, budget_tenths, max_per_club, incumbent):
    """Exact branch and bound for one formation. Returns (value, labels) or None.

    `budget_tenths` is an integer number of £0.1m units, or None for unlimited.
    """
    counts = [needs[position] for position in POSITIONS]
    if any(pools[i].size < counts[i] for i in range(len(POSITIONS))):
        return None                       # e.g. a blank gameweek

    # Unconstrained best / cheapest completion of every LATER position.
    best_after = [0.0] * (len(POSITIONS) + 1)
    cost_after = [0] * (len(POSITIONS) + 1)
    for i in range(len(POSITIONS) - 1, -1, -1):
        best_after[i] = best_after[i + 1] + pools[i].best_values(counts[i])
        cost_after[i] = cost_after[i + 1] + pools[i].cheapest_fill(counts[i])

    unlimited = budget_tenths is None
    # node_bound[p][k][b]: k more here + all later positions, sharing `b` optimally.
    node_bound = None if unlimited else _node_bounds(pools, counts, budget_tenths)

    best = {"value": incumbent, "labels": None}
    clubs_used: Dict[object, int] = {}
    chosen: List[Tuple[int, int]] = []

    def fill(position_index, start, remaining, value, cost):
        if remaining == 0:
            if position_index + 1 == len(POSITIONS):
                if value > best["value"]:
                    best["value"] = value
                    best["labels"] = [pools[p].labels[i] for p, i in chosen]
                return
            return fill(position_index + 1, 0, counts[position_index + 1], value, cost)

        pool = pools[position_index]
        # Cheapest legal completion from here: k cheapest in this pool + later pools.
        floor_cost = cost_after[position_index + 1]
        if not unlimited:
            if cost + pool.min_cost[remaining][start] + floor_cost > budget_tenths:
                return                    # only worsens as `start` advances

            # The most this node could still reach, with the money actually left.
            left = budget_tenths - cost
            reachable = value + node_bound[position_index][remaining][left]
            if reachable <= best["value"] + BOUND_TOLERANCE:
                return

        limit = pool.size - remaining
        for index in range(start, limit + 1):
            ceiling = (
                value
                + float(pool.values[index:index + remaining].sum())
                + best_after[position_index + 1]
            )
            if ceiling <= best["value"] + BOUND_TOLERANCE:
                return                    # values descend: no later index can beat it

            club = pool.clubs[index]
            if clubs_used.get(club, 0) >= max_per_club:
                continue

            price = pool.prices[index]
            if not unlimited:
                rest = pool.min_cost[remaining - 1][index + 1] + floor_cost
                if cost + price + rest > budget_tenths:
                    continue              # he is unaffordable; a cheaper one may not be

            clubs_used[club] = clubs_used.get(club, 0) + 1
            chosen.append((position_index, index))
            fill(position_index, index + 1, remaining - 1,
                 value + float(pool.values[index]), cost + price)
            chosen.pop()
            clubs_used[club] -= 1

    fill(0, 0, counts[0], 0.0, 0)
    return None if best["labels"] is None else (best["value"], best["labels"])


def select_xi(
    players: pd.DataFrame,
    value_column: str,
    budget: Optional[float] = None,
    max_per_club: int = MAX_PER_CLUB,
) -> Optional[XiSelection]:
    """The provably highest-`value_column` legal XI, or None if none exists.

    `players` needs columns `position` (1-4), `club`, `price` and `value_column`.
    Ties are broken by whichever player the sort happened to put first; when two
    XIs have identical projected value they are interchangeable by construction.
    """
    for column in ("position", "club", "price", value_column):
        if column not in players.columns:
            raise ValueError("players is missing required column '%s'" % column)
    if len(players) < SQUAD_SIZE:
        return None

    candidates = drop_dominated(players, value_column, max_per_club)
    pools = [_PositionPool(candidates[candidates["position"] == position], value_column)
             for position in POSITIONS]
    budget_tenths = None if budget is None else _to_tenths(budget)

    # If the unconstrained best XI is affordable anyway, it is also the budgeted
    # best XI - a budget only removes options, and this option survives. Skips the
    # expensive bound tables whenever money is not actually the binding constraint.
    if budget_tenths is not None:
        unlimited = _search(players, pools, value_column, None, max_per_club)
        if unlimited is not None and _to_tenths(unlimited.cost) <= budget_tenths:
            return unlimited

        for pool in pools:
            pool.build_affordable_table(budget_tenths)

    return _search(players, pools, value_column, budget_tenths, max_per_club)


def _search(players, pools, value_column, budget_tenths, max_per_club) -> Optional[XiSelection]:
    best_value, best_labels, best_formation = -np.inf, None, None

    # Seed with a good legal XI so the bound has something to prune against.
    for formation in FORMATIONS:
        counts = [_formation_needs(formation)[position] for position in POSITIONS]
        if any(pools[i].size < counts[i] for i in range(len(POSITIONS))):
            continue
        seed = _greedy_xi(pools, counts, budget_tenths, max_per_club)
        if seed is not None and seed[0] > best_value:
            best_value, best_labels = seed
            best_formation = formation

    for formation in FORMATIONS:
        result = _solve_formation(
            pools, _formation_needs(formation), budget_tenths, max_per_club, best_value)
        if result is not None:
            best_value, best_labels = result
            best_formation = formation

    if best_labels is None:
        return None

    squad = players.loc[best_labels]
    captain = squad[value_column].idxmax()
    return XiSelection(
        labels=list(best_labels),
        projected=float(best_value),
        cost=float(squad["price"].sum()),
        formation=best_formation,
        captain=captain,
    )


def xi_actual_points(
    players: pd.DataFrame,
    selection,
    actual_column: str = "actual",
    with_captain: bool = False,
) -> float:
    """What the selected XI really scored. The captain's score counts twice.

    Accepts an `XiSelection` or a `SquadSelection`; only the starting eleven scores,
    which is the FPL rule and the reason a bench is worth nothing but money.
    """
    starters = selection.xi if isinstance(selection, SquadSelection) else selection.labels
    squad = players.loc[starters]
    total = float(squad[actual_column].sum())
    if with_captain:
        total += float(players.loc[selection.captain, actual_column])
    return total


# ---------------------------------------------------------------------------
# The real game: a 15-man squad for 100.0m, of which eleven start.
# ---------------------------------------------------------------------------

SquadSelection = namedtuple("SquadSelection", "xi bench projected cost formation captain")


def _bench_needs(formation: Tuple[int, int, int]) -> Dict[int, int]:
    """What the bench is FORCED to contain, given the XI's formation."""
    xi = _formation_needs(formation)
    return {position: SQUAD_QUOTA[position] - xi[position] for position in POSITIONS}


class _BenchPool(object):
    """Per position, candidates cheapest-first, with prefix sums.

    Because the list is sorted by price, the cheapest `k` players from index `i` are
    exactly `candidates[i:i+k]` - so a prefix sum gives the cheapest possible
    completion in constant time. That bound is what stops the bench search from
    enumerating the league: without it, 13 bench searches ran the inner loop 23.9
    million times and took 42 of 42.6 seconds.
    """

    __slots__ = ("prices", "clubs", "labels", "prefix", "size")

    def __init__(self, rows: pd.DataFrame):
        ordered = sorted(
            (_to_tenths(price), club, label)
            for label, club, price in zip(rows.index, rows["club"], rows["price"])
        )
        self.prices = [price for price, _, _ in ordered]
        self.clubs = [club for _, club, _ in ordered]
        self.labels = [label for _, _, label in ordered]
        self.size = len(ordered)
        self.prefix = [0] * (self.size + 1)
        for i, price in enumerate(self.prices):
            self.prefix[i + 1] = self.prefix[i] + price

    def cheapest_from(self, start: int, count: int) -> int:
        """Cheapest total price of `count` players from `start` onward, ignoring the
        club cap and exclusions - so it only ever under-states. UNAFFORDABLE if the
        suffix is too short."""
        if start + count > self.size:
            return UNAFFORDABLE
        return self.prefix[start + count] - self.prefix[start]


def _price_sorted(players: pd.DataFrame) -> Dict[int, _BenchPool]:
    return {position: _BenchPool(players[players["position"] == position])
            for position in POSITIONS}


def _cheapest_bench(price_sorted, needs, excluded, clubs_used, max_per_club, ceiling):
    """The cheapest legal bench, exactly. Returns (cost_tenths, labels) or None.

    Not simply "the cheapest player in each slot": FPL's 3-per-club limit spans the
    whole squad, so the cheapest bench keeper may be unusable because his club
    already has three starters, and the eleven have first claim on everyone.

    Exact branch and bound over at most four slots. Two bounds, both computed from
    prefix sums and both ignoring the club cap and exclusions - which means they
    under-state the true cost, so they can never prune a bench that was affordable:

        ceiling   the money left before the squad busts its budget
        incumbent the cheapest complete bench found so far
    """
    positions = [p for p in POSITIONS if needs[p] > 0]
    # Cheapest conceivable fill of every LATER position.
    later_floor = [0] * (len(positions) + 1)
    for i in range(len(positions) - 1, -1, -1):
        position = positions[i]
        later_floor[i] = later_floor[i + 1] + price_sorted[position].cheapest_from(
            0, needs[position])

    best = {"cost": None, "labels": None}
    picked: List[object] = []

    def fill_position(index, cost, clubs):
        if index == len(positions):
            if best["cost"] is None or cost < best["cost"]:
                best["cost"], best["labels"] = cost, list(picked)
            return

        position = positions[index]
        pool = price_sorted[position]
        rest = later_floor[index + 1]

        def take(start, remaining, cost):
            if remaining == 0:
                return fill_position(index + 1, cost, clubs)

            floor = cost + pool.cheapest_from(start, remaining) + rest
            if floor > ceiling:
                return                       # only dearer as `start` advances
            if best["cost"] is not None and floor >= best["cost"]:
                return

            for offset in range(start, pool.size - remaining + 1):
                # Cheapest completion that takes THIS player: he plus the next ones.
                floor = (cost + pool.prices[offset]
                         + pool.cheapest_from(offset + 1, remaining - 1) + rest)
                if floor > ceiling:
                    return                   # prices ascend: no later offset is cheaper
                if best["cost"] is not None and floor >= best["cost"]:
                    return

                label = pool.labels[offset]
                club = pool.clubs[offset]
                if label in excluded or clubs.get(club, 0) >= max_per_club:
                    continue

                clubs[club] = clubs.get(club, 0) + 1
                picked.append(label)
                take(offset + 1, remaining - 1, cost + pool.prices[offset])
                picked.pop()
                clubs[club] -= 1

        take(0, needs[position], cost)

    fill_position(0, 0, dict(clubs_used))
    return None if best["labels"] is None else (best["cost"], best["labels"])


def _bench_floor(price_sorted, needs) -> int:
    """A lower bound on any bench's cost: the cheapest players of each required
    position, ignoring the club cap and ignoring that the XI may want them.
    Only ever under-states, so pruning with it is safe."""
    return sum(price_sorted[position].cheapest_from(0, count)
               for position, count in needs.items() if count)


def select_squad(
    players: pd.DataFrame,
    value_column: str,
    squad_budget: float = TOTAL_SQUAD_BUDGET,
    max_per_club: int = MAX_PER_CLUB,
) -> Optional[SquadSelection]:
    """The provably best legal FPL squad: 15 players for `squad_budget`, of which
    the eleven that start maximise `value_column`.

    The bench contributes no points, so it exists only to be cheap and legal. Its
    composition is dictated by the formation (the squad quota is fixed at 2/5/5/3),
    which is precisely why an "XI budget" is not a real quantity.

    Search: branch and bound over the XI, exactly as `select_xi`, but every cost
    bound reserves `_bench_floor` for the four players still to be bought, and each
    complete XI is only accepted once a genuine cheapest legal bench is found for
    it. The floor under-states the bench, so no affordable squad is ever pruned.
    """
    for column in ("position", "club", "price", value_column):
        if column not in players.columns:
            raise ValueError("players is missing required column '%s'" % column)
    for position, quota in SQUAD_QUOTA.items():
        if (players["position"] == position).sum() < quota:
            return None

    budget_tenths = _to_tenths(squad_budget)
    candidates = drop_dominated(players, value_column, max_per_club)
    for position, quota in SQUAD_QUOTA.items():
        if (candidates["position"] == position).sum() < quota:
            candidates = players            # reduction cannot starve a quota; bail out
            break

    price_sorted = _price_sorted(candidates)
    pools = [_PositionPool(candidates[candidates["position"] == position], value_column)
             for position in POSITIONS]
    for pool in pools:
        pool.build_affordable_table(budget_tenths)

    best = {"value": -np.inf, "xi": None, "bench": None, "cost": 0, "formation": None}
    for formation in FORMATIONS:
        _seed_squad(pools, price_sorted, formation, budget_tenths, max_per_club, best)
    for formation in FORMATIONS:
        _solve_squad_formation(pools, price_sorted, formation,
                               budget_tenths, max_per_club, best)

    if best["xi"] is None:
        return None
    squad = players.loc[best["xi"]]
    return SquadSelection(
        xi=list(best["xi"]),
        bench=list(best["bench"]),
        projected=float(best["value"]),
        cost=best["cost"] / float(TENTHS_PER_MILLION),
        formation=best["formation"],
        captain=squad[value_column].idxmax(),
    )


def _seed_squad(pools, price_sorted, formation, budget_tenths, max_per_club, best):
    """A decent legal SQUAD, found fast, purely to seed the search.

    Same reason as `_greedy_xi`: with no incumbent the bound prunes nothing and the
    first dive chases an unaffordable dream team. Measured, this is the difference
    between ~40 seconds and well under one. The seed must be genuinely feasible -
    an infeasible incumbent could prune the real optimum - so its bench is bought
    for real before it is accepted.
    """
    counts = [_formation_needs(formation)[position] for position in POSITIONS]
    if any(pools[i].size < counts[i] for i in range(len(POSITIONS))):
        return
    bench_needs = _bench_needs(formation)
    xi_budget = budget_tenths - _bench_floor(price_sorted, bench_needs)
    if xi_budget < 0:
        return

    seed = _greedy_xi(pools, counts, xi_budget, max_per_club)
    if seed is None:
        return
    value, labels = seed
    if value <= best["value"]:
        return

    clubs, cost = {}, 0
    lookup = {}
    for index, pool in enumerate(pools):
        for position_index, label in enumerate(pool.labels):
            lookup[label] = (index, position_index)
    for label in labels:
        pool_index, position_index = lookup[label]
        clubs[pools[pool_index].clubs[position_index]] = clubs.get(
            pools[pool_index].clubs[position_index], 0) + 1
        cost += pools[pool_index].prices[position_index]

    bench = _cheapest_bench(price_sorted, bench_needs, set(labels), clubs,
                            max_per_club, budget_tenths - cost)
    if bench is None:
        return
    bench_cost, bench_labels = bench
    best.update(value=value, xi=labels, bench=bench_labels,
                cost=cost + bench_cost, formation=formation)


def _solve_squad_formation(pools, price_sorted, formation,
                           budget_tenths, max_per_club, best):
    counts = [_formation_needs(formation)[position] for position in POSITIONS]
    if any(pools[i].size < counts[i] for i in range(len(POSITIONS))):
        return

    bench_needs = _bench_needs(formation)
    floor = _bench_floor(price_sorted, bench_needs)
    xi_budget = budget_tenths - floor
    if xi_budget < 0:
        return

    cost_after = [0] * (len(POSITIONS) + 1)
    best_after = [0.0] * (len(POSITIONS) + 1)
    for i in range(len(POSITIONS) - 1, -1, -1):
        cost_after[i] = cost_after[i + 1] + pools[i].cheapest_fill(counts[i])
        best_after[i] = best_after[i + 1] + pools[i].best_values(counts[i])
    if cost_after[0] > xi_budget:
        return

    node_bound = _node_bounds(pools, counts, xi_budget)
    clubs_used: Dict[object, int] = {}
    chosen: List[Tuple[int, int]] = []

    def accept(value, cost):
        """A complete XI. It is only real if a legal bench can be bought for it."""
        xi_labels = [pools[p].labels[i] for p, i in chosen]
        bench = _cheapest_bench(price_sorted, bench_needs, set(xi_labels),
                                clubs_used, max_per_club, budget_tenths - cost)
        if bench is None:
            return                       # this XI cannot be benched legally: not a squad
        bench_cost, bench_labels = bench
        best.update(value=value, xi=xi_labels, bench=bench_labels,
                    cost=cost + bench_cost, formation=formation)

    def fill(position_index, start, remaining, value, cost):
        if remaining == 0:
            if position_index + 1 == len(POSITIONS):
                if value > best["value"]:
                    accept(value, cost)
                return
            return fill(position_index + 1, 0, counts[position_index + 1], value, cost)

        pool = pools[position_index]
        if cost + pool.min_cost[remaining][start] + cost_after[position_index + 1] > xi_budget:
            return
        left = xi_budget - cost
        if value + node_bound[position_index][remaining][left] <= best["value"] + BOUND_TOLERANCE:
            return

        limit = pool.size - remaining
        for index in range(start, limit + 1):
            ceiling = (value + float(pool.values[index:index + remaining].sum())
                       + best_after[position_index + 1])
            if ceiling <= best["value"] + BOUND_TOLERANCE:
                return

            club = pool.clubs[index]
            if clubs_used.get(club, 0) >= max_per_club:
                continue
            price = pool.prices[index]
            rest = pool.min_cost[remaining - 1][index + 1] + cost_after[position_index + 1]
            if cost + price + rest > xi_budget:
                continue

            clubs_used[club] = clubs_used.get(club, 0) + 1
            chosen.append((position_index, index))
            fill(position_index, index + 1, remaining - 1,
                 value + float(pool.values[index]), cost + price)
            chosen.pop()
            clubs_used[club] -= 1

    fill(0, 0, counts[0], 0.0, 0)
