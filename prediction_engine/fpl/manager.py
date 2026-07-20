"""Carry a squad across gameweeks and decide transfers — the game as actually played.

Everything else in this package answers "how good is the projection?". It answers it
by rebuilding a fresh £100m squad every gameweek, which is the right way to isolate
projection quality (`benchmark_fpl_optimizer` explains why) and the wrong way to play
FPL. You buy a squad ONCE. After that you get one free transfer a week, and every
extra one costs four points.

So this module is a different question, and it must not be confused with the first:

    projection quality   given a free rebuild, do our numbers beat theirs?   PROVEN
    manager quality      given a carried squad, is acting better than holding?  OPEN

Both run on the same projections. The second is measured against `hold` — buy the
opening squad and never touch it — because that is the honest null. A transfer policy
that cannot beat doing nothing has no business shipping, and "it obviously helps" is
exactly the kind of folklore this project exists to kill.

The rules modelled here are the real ones:

    selling price   you do NOT get the current price back. Profit is halved and
                    rounded down to 0.1m; losses are yours in full. Ignoring this
                    silently hands the manager money the game never gives him.
    free transfers  one a week, banked up to a cap that CHANGED in 2024/25 (2 -> 5),
                    so it is a per-season rule, not a constant.
    hits            every transfer beyond the free ones costs 4 points, deducted
                    from that gameweek's score.
    quota + caps    a transfer is position-for-position (the 2/5/5/3 quota is fixed)
                    and must respect the 3-per-club limit across all fifteen.

Deliberate v1 limits, stated rather than buried:

  - HORIZON 1. A transfer is judged on the NEXT gameweek only. A multi-week horizon
    is the obvious improvement, but in a backtest it is also the obvious way to cheat:
    the cached frame's gameweek k+1 projection was computed with data up to k+1's
    deadline, so consulting it at k's deadline leaks the result of gameweek k. Doing
    it properly means re-projecting future fixtures from k's information set. That is
    a later version, and it must be built before it is measured, not after.
  - NO CHIPS. Wildcard, free hit, bench boost and triple captain are each a separate
    decision problem layered on top of this one.
  - NO AUTOSUBS, matching `scorer` and the benchmarks, so the comparison is unaffected.
  - Sequential-greedy multi-transfer search: the best single transfer, then the best
    one after it. Not jointly optimal for 2+ transfers, and not claimed to be.

The captain is doubled and hits are subtracted, so a gameweek score here is what the
game would actually have shown you.
"""

import logging
from collections import namedtuple
from typing import Dict, List, Optional, Sequence, Tuple

from prediction_engine.fpl.optimizer import (FORMATIONS, MAX_PER_CLUB, POSITIONS,
                                             SQUAD_QUOTA, TENTHS_PER_MILLION,
                                             TOTAL_SQUAD_BUDGET, _to_tenths)
from prediction_engine.fpl.scoring import DEF, FWD, GKP, MID

logger = logging.getLogger(__name__)

HIT_COST = 4                    # points deducted per transfer beyond the free ones
TRANSFERS_EARNED_PER_GAMEWEEK = 1

# The cap on banked free transfers is NOT a constant: FPL raised it from 2 to 5 for
# 2024/25. A backtest that applies today's cap to 2022/23 is measuring a game nobody
# played, so the rule is looked up by season.
FREE_TRANSFER_CAP_BY_SEASON = {
    "2018-19": 2, "2019-20": 2, "2020-21": 2, "2021-22": 2,
    "2022-23": 2, "2023-24": 2,
    "2024-25": 5, "2025-26": 5, "2026-27": 5,
}
DEFAULT_FREE_TRANSFER_CAP = 5   # the current rule, for seasons not listed above


def free_transfer_cap(season: Optional[str]) -> int:
    """The banked-free-transfer cap in force for `season`."""
    if season is None:
        return DEFAULT_FREE_TRANSFER_CAP
    return FREE_TRANSFER_CAP_BY_SEASON.get(str(season), DEFAULT_FREE_TRANSFER_CAP)


def selling_price_tenths(bought_tenths: int, current_tenths: int) -> int:
    """What FPL actually pays you for a player, in tenths of a million.

    Profit is halved and rounded DOWN to the nearest 0.1m; a fall in price is borne
    in full. So a player bought at 7.0 now worth 7.5 sells for 7.2, not 7.5 — and one
    bought at 7.0 now worth 6.5 sells for 6.5, not 6.7.

    Getting this wrong in the manager's favour would quietly inflate every transfer
    policy's budget relative to the `hold` baseline it is being judged against, which
    is precisely the sort of free money that manufactures a fake edge.
    """
    rise = current_tenths - bought_tenths
    if rise <= 0:
        return current_tenths
    return bought_tenths + rise // 2


XiChoice = namedtuple("XiChoice", "xi bench captain projected")


def best_xi(squad_ids: Sequence[int], projections: Dict[int, float],
            positions: Dict[int, int]) -> Optional[XiChoice]:
    """The best legal XI from a fixed 15-man squad, exactly, and its captain.

    Exact by construction rather than by search: the squad already satisfies the club
    cap and there is no budget left to spend, so the only constraint is the formation.
    Within a position you always want the highest-projected players, so for each of
    the eight legal shapes the best XI is just the top `k` of each position — and the
    best overall is the best of those eight. No branch and bound needed.

    `projected` counts the captain twice, because that is what the gameweek is worth.
    """
    by_position: Dict[int, List[Tuple[float, int]]] = {p: [] for p in POSITIONS}
    for pid in squad_ids:
        position = positions.get(int(pid))
        if position is None:
            continue
        by_position[int(position)].append((float(projections.get(int(pid), 0.0)), int(pid)))
    for position in POSITIONS:
        # Descending projection; player id breaks ties so the choice is deterministic
        # across runs and platforms rather than dependent on dict ordering.
        by_position[position].sort(key=lambda pair: (-pair[0], pair[1]))

    best: Optional[XiChoice] = None
    for defenders, midfielders, forwards in FORMATIONS:
        needs = {GKP: 1, DEF: defenders, MID: midfielders, FWD: forwards}
        if any(len(by_position[p]) < needs[p] for p in POSITIONS):
            continue
        xi: List[int] = []
        total = 0.0
        for position in POSITIONS:
            for value, pid in by_position[position][:needs[position]]:
                xi.append(pid)
                total += value
        captain = max(xi, key=lambda pid: (float(projections.get(pid, 0.0)), -pid))
        total += float(projections.get(captain, 0.0))       # captaincy: doubled
        if best is None or total > best.projected:
            bench = [pid for pid in squad_ids if int(pid) not in set(xi)]
            best = XiChoice(xi=xi, bench=bench, captain=captain, projected=total)
    return best


def squad_projection(squad_ids, projections, positions) -> float:
    """Projected points of the best XI this squad can field, captain doubled."""
    choice = best_xi(squad_ids, projections, positions)
    return 0.0 if choice is None else choice.projected


class Squad:
    """A carried FPL squad: fifteen players, a bank, and banked free transfers.

    Prices are held in integer tenths throughout, for the same reason the optimizer
    does: a squad costing exactly £100.0m must not be rejected by float noise.

    `bought` records what was PAID for each player, which the selling-price rule
    needs. It is not the same as what he is worth now, and conflating the two is the
    single easiest way to give the manager money he never had.
    """

    def __init__(self, player_ids: Sequence[int], bought_tenths: Dict[int, int],
                 bank_tenths: int = 0, free_transfers: int = 1,
                 season: Optional[str] = None):
        self.players = [int(pid) for pid in player_ids]
        self.bought = {int(pid): int(price) for pid, price in bought_tenths.items()}
        self.bank = int(bank_tenths)
        self.free_transfers = int(free_transfers)
        self.season = season
        self.cap = free_transfer_cap(season)

    def copy(self) -> "Squad":
        clone = Squad(list(self.players), dict(self.bought), self.bank,
                      self.free_transfers, self.season)
        return clone

    def contains(self, player_id: int) -> bool:
        return int(player_id) in set(self.players)

    def club_counts(self, clubs: Dict[int, object]) -> Dict[object, int]:
        counts: Dict[object, int] = {}
        for pid in self.players:
            club = clubs.get(pid)
            counts[club] = counts.get(club, 0) + 1
        return counts

    def value_tenths(self, prices: Dict[int, int]) -> int:
        """Squad value at SELLING prices plus the bank — what you could actually
        raise. Not the same as the sum of current prices."""
        total = self.bank
        for pid in self.players:
            current = prices.get(pid, self.bought.get(pid, 0))
            total += selling_price_tenths(self.bought.get(pid, current), current)
        return total

    def apply_transfer(self, out_id: int, in_id: int, in_price_tenths: int,
                       out_price_tenths: int) -> None:
        """Execute one transfer, moving money correctly. Assumes legality is checked."""
        proceeds = selling_price_tenths(self.bought.get(int(out_id), out_price_tenths),
                                        out_price_tenths)
        self.players.remove(int(out_id))
        self.bought.pop(int(out_id), None)
        self.players.append(int(in_id))
        self.bought[int(in_id)] = int(in_price_tenths)
        self.bank += proceeds - int(in_price_tenths)

    def award_free_transfer(self) -> None:
        """One a week, banked up to the season's cap."""
        self.free_transfers = min(self.cap,
                                  self.free_transfers + TRANSFERS_EARNED_PER_GAMEWEEK)


TransferPlan = namedtuple("TransferPlan", "moves hits gain")


def best_single_transfer(squad: Squad, projections: Dict[int, float],
                         prices: Dict[int, int], clubs: Dict[int, object],
                         positions: Dict[int, int], candidates: Sequence[int],
                         max_per_club: int = MAX_PER_CLUB):
    """The transfer that most improves the projected XI, or None if none improves it.

    The objective is the change in the SQUAD's projected XI score (captain included),
    not the change in the two players' projections. That distinction is the whole
    point: upgrading a bench player who still will not start is worth exactly zero,
    and a naive "buy the highest projection I can afford" policy spends real money on
    exactly that.

    Exact, and it does NOT evaluate every (out, in) pair. For a fixed player sold, the
    squad's projected XI is a non-decreasing function of the incoming player's
    projection — he enters the same position, so a higher number can only ever help
    the best XI and never hurt it. Therefore the best incoming player for that slot is
    simply the highest-projected LEGAL, AFFORDABLE one, and the remaining candidates
    cannot beat him. That turns ~15x600 squad evaluations a week into 15, which is the
    difference between a backtest that runs and one that does not.
    """
    baseline = squad_projection(squad.players, projections, positions)
    # Highest projection first; cheaper wins ties (money is next week's optionality),
    # then id, so a run is reproducible rather than dict-order dependent.
    ordered = sorted((int(pid) for pid in candidates),
                     key=lambda pid: (-float(projections.get(pid, 0.0)),
                                      prices.get(pid, 0), pid))
    held = set(int(pid) for pid in squad.players)
    counts = squad.club_counts(clubs)
    best = None

    for out_id in list(squad.players):
        out_id = int(out_id)
        position = positions.get(out_id)
        if position is None:
            continue
        out_price = prices.get(out_id, squad.bought.get(out_id, 0))
        proceeds = selling_price_tenths(squad.bought.get(out_id, out_price), out_price)
        budget = squad.bank + proceeds
        out_club = clubs.get(out_id)

        target = None
        for pid in ordered:
            if pid in held or positions.get(pid) != position:
                continue
            price = prices.get(pid)
            if price is None or price > budget:
                continue
            # The outgoing player's club slot is freed before the incoming one takes it.
            club = clubs.get(pid)
            used = counts.get(club, 0) - (1 if club == out_club else 0)
            if used >= max_per_club:
                continue
            target = pid
            break                       # sorted by projection: none later can beat him
        if target is None:
            continue

        remaining = [pid for pid in squad.players if int(pid) != out_id]
        gain = squad_projection(remaining + [target], projections, positions) - baseline
        if gain <= 0:
            continue
        key = (gain, -prices[target], -target)
        if best is None or key > best[0]:
            best = (key, gain, out_id, target, prices[target], out_price)

    if best is None:
        return None
    _, gain, out_id, in_id, in_price, out_price = best
    return {"out": out_id, "in": in_id, "in_price": in_price,
            "out_price": out_price, "gain": gain}


def plan_transfers(squad: Squad, projections, prices, clubs, positions, candidates,
                   max_transfers: int = 2, hit_threshold: float = float("inf"),
                   free_threshold: float = 0.0,
                   max_per_club: int = MAX_PER_CLUB) -> TransferPlan:
    """Decide this week's transfers. Returns the moves, the points they cost, and the
    projected gain — WITHOUT mutating `squad`.

    Two thresholds, because the two decisions are genuinely different:

        free_threshold  a free transfer is not free. Using it today means not having
                        it next week, so a marginal upgrade can be worth less than
                        the option it burns. This is the minimum projected gain that
                        justifies spending one.
        hit_threshold   a paid transfer must clear 4 points to break even on the week
                        alone. Set it to infinity (the default) to forbid hits, which
                        is the conservative policy and the one to beat first.

    Sequential greedy: take the best single transfer, then the best one after it.
    Not jointly optimal for two or more, and not claimed to be — a joint search over
    pairs is ~9,000² evaluations a week and can wait until one transfer is proven to
    be worth making at all.
    """
    working = squad.copy()
    moves: List[dict] = []
    total_gain = 0.0
    hits = 0

    for index in range(max_transfers):
        move = best_single_transfer(working, projections, prices, clubs, positions,
                                    candidates, max_per_club)
        if move is None:
            break
        is_free = index < working.free_transfers
        threshold = free_threshold if is_free else hit_threshold
        if move["gain"] <= threshold:
            break
        working.apply_transfer(move["out"], move["in"], move["in_price"],
                               move["out_price"])
        moves.append(move)
        total_gain += move["gain"]
        if not is_free:
            hits += 1

    return TransferPlan(moves=moves, hits=hits * HIT_COST, gain=total_gain)


def execute(squad: Squad, plan: TransferPlan) -> None:
    """Apply a plan to a squad and settle the free-transfer ledger."""
    for move in plan.moves:
        squad.apply_transfer(move["out"], move["in"], move["in_price"],
                             move["out_price"])
    used = len(plan.moves)
    squad.free_transfers = max(0, squad.free_transfers - used)


GameweekResult = namedtuple(
    "GameweekResult",
    "gameweek points hits net xi captain transfers squad_value bank free_transfers")


def score_gameweek(squad: Squad, projections, actuals, positions, prices,
                   gameweek: int, plan: Optional[TransferPlan] = None) -> GameweekResult:
    """What this squad actually scored: best projected XI, captain doubled, hits off.

    The XI is chosen by PROJECTION and scored by ACTUALS — never the other way round,
    which would be hindsight and would flatter every policy identically and uselessly.
    """
    choice = best_xi(squad.players, projections, positions)
    if choice is None:
        return GameweekResult(gameweek, 0.0, 0.0, 0.0, [], None, 0,
                              squad.value_tenths(prices) / float(TENTHS_PER_MILLION),
                              squad.bank / float(TENTHS_PER_MILLION),
                              squad.free_transfers)
    points = sum(float(actuals.get(int(pid), 0.0)) for pid in choice.xi)
    points += float(actuals.get(int(choice.captain), 0.0))       # doubled
    hits = 0.0 if plan is None else float(plan.hits)
    return GameweekResult(
        gameweek=int(gameweek),
        points=float(points),
        hits=hits,
        net=float(points) - hits,
        xi=list(choice.xi),
        captain=int(choice.captain),
        transfers=0 if plan is None else len(plan.moves),
        squad_value=squad.value_tenths(prices) / float(TENTHS_PER_MILLION),
        bank=squad.bank / float(TENTHS_PER_MILLION),
        free_transfers=squad.free_transfers,
    )


def opening_squad(week_frame, value_column: str,
                  budget: float = TOTAL_SQUAD_BUDGET,
                  season: Optional[str] = None) -> Optional[Squad]:
    """Buy the opening squad with the existing exact optimizer.

    Every policy starts from the IDENTICAL squad, so a difference between them later
    is a difference in transfer decisions and nothing else.
    """
    from prediction_engine.fpl.optimizer import select_squad

    selection = select_squad(week_frame, value_column, squad_budget=budget)
    if selection is None:
        return None
    labels = list(selection.xi) + list(selection.bench)
    ids = [int(week_frame.loc[label, "player_id"]) for label in labels]
    bought = {int(week_frame.loc[label, "player_id"]):
              _to_tenths(week_frame.loc[label, "price"]) for label in labels}
    spent = sum(bought.values())
    return Squad(ids, bought, bank_tenths=_to_tenths(budget) - spent,
                 free_transfers=1, season=season)
