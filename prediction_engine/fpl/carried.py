"""The live carried-squad tracks: let the season settle what the backtest could not.

`benchmark_fpl_transfers` proved that ACTING beats HOLDING (+7.17 net pts/GW) and
failed to prove that our projections choose better transfers than a season average
(-1.56/GW, t p=0.31 — a null, directionally negative). The second result is the one
that decides how the live team should actually be run, and 131 backtest gameweeks
were not enough to settle it.

So it goes to the season, as a paired A/B, in exactly the form the backtest ran:

    carried_ours   the squad, maintained by OUR projections
    carried_ppg    the SAME squad, maintained by `player_ppg`

Both tracks start from the identical opening squad — the pre-registered primary's —
so from the first transfer onward the only thing that differs between them is which
number chose the transfer. That is what makes the comparison worth anything.

Neither track is the pre-registered primary endpoint. The primary rebuilds a fresh
£100m squad every gameweek and MUST NOT change: it is the thing the 8-season proof
describes, and rewriting it now would void the pre-registration that makes any live
result believable. These are declared variants, scored separately, exactly like the
`lineups` candidate.

INTEGRITY. A carried squad is stateful, which is a hazard the rebuild endpoint does
not have: if the state could be edited after kickoff, the whole track would be
worthless. Two controls:

  - state is written to `research/results/live/<season>/` BEFORE the deadline and
    committed, the same narrow gitignore exception the artifacts use, so the record
    is in git history with a timestamp rather than sitting on my disk;
  - every step records the gameweek it was taken for and refuses to run twice for
    the same gameweek, so a rerun cannot silently re-decide a transfer with
    information that arrived after the deadline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from prediction_engine.fpl.manager import (Squad, best_xi, execute, plan_transfers)
from prediction_engine.fpl.optimizer import TENTHS_PER_MILLION, _to_tenths

logger = logging.getLogger(__name__)

LIVE_DIR = Path(__file__).resolve().parents[2] / "research" / "results" / "live"

# The two tracks and the column each uses to choose transfers. `carried_ours` is the
# candidate; `carried_ppg` is the control it has to beat.
TRACKS = {
    "carried_ours": "expected_points",
    "carried_ppg": "player_ppg",
}

# The policy fixed by the backtest's pre-specified primary: one free transfer a week,
# made whenever it improves the projected XI at all, never taking a hit. Named here so
# the live tracks run the SAME policy that was measured, rather than a tuned variant
# nobody tested.
POLICY = dict(max_transfers=1, free_threshold=0.0, hit_threshold=float("inf"))


def state_path(season: str, track: str) -> Path:
    return LIVE_DIR / str(season) / ("%s.json" % track)


def load_state(season: str, track: str) -> Optional[dict]:
    path = state_path(season, track)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(season: str, track: str, state: dict) -> Path:
    path = state_path(season, track)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    logger.info("wrote %s state for GW%s to %s", track, state.get("gameweek"), path)
    return path


def _squad_from_state(state: dict, season: str) -> Squad:
    return Squad(
        [int(pid) for pid in state["players"]],
        {int(pid): int(price) for pid, price in state["bought"].items()},
        bank_tenths=int(state["bank"]),
        free_transfers=int(state["free_transfers"]),
        season=season,
    )


def _state_from_squad(squad: Squad, season: str, gameweek: int,
                      moves: List[dict], history: List[dict]) -> dict:
    return {
        "season": str(season),
        "gameweek": int(gameweek),
        "players": [int(pid) for pid in squad.players],
        "bought": {str(pid): int(price) for pid, price in squad.bought.items()},
        "bank": int(squad.bank),
        "free_transfers": int(squad.free_transfers),
        "history": history + [{"gameweek": int(gameweek), "moves": moves}],
    }


def _maps(frame: pd.DataFrame, value_column: str):
    """Per-gameweek lookups keyed by FPL player id, from a bank_it gameweek frame."""
    ids = frame["player_id"].astype(int)
    return {
        "projections": dict(zip(ids, frame[value_column].astype(float))),
        "prices": {int(pid): _to_tenths(price) for pid, price in zip(ids, frame["price"])},
        "clubs": dict(zip(ids, frame["club"])),
        "positions": {int(pid): int(pos) for pid, pos in zip(ids, frame["position_id"])},
    }


def open_track(frame: pd.DataFrame, opening_squad_ids: List[int], season: str,
               gameweek: int, track: str) -> dict:
    """Start a track from the pre-registered primary's squad.

    Both tracks open from the SAME fifteen, which is what turns the season into a
    clean paired test of the transfer decision rather than a muddle of two different
    squads drifting apart from the start.
    """
    prices = {int(pid): _to_tenths(price)
              for pid, price in zip(frame["player_id"].astype(int), frame["price"])}
    bought = {int(pid): prices[int(pid)] for pid in opening_squad_ids}
    budget = _to_tenths(100.0)
    squad = Squad(opening_squad_ids, bought,
                  bank_tenths=budget - sum(bought.values()),
                  free_transfers=1, season=season)
    logger.info("opened %s at GW%d with %d players, bank %.1f",
                track, gameweek, len(squad.players), squad.bank / 10.0)
    return _state_from_squad(squad, season, gameweek, moves=[], history=[])


def step(frame: pd.DataFrame, state: dict, track: str, season: str,
         gameweek: int) -> dict:
    """Advance one track by one gameweek: award the free transfer, decide, execute.

    Refuses to re-run a gameweek that has already been decided. A carried squad is
    stateful, so a silent second run would let a transfer be re-chosen with
    information that arrived after the deadline — which is the one thing this whole
    apparatus exists to prevent.
    """
    if int(state["gameweek"]) >= int(gameweek):
        raise ValueError(
            "%s is already at GW%s; refusing to re-decide GW%d. Delete the state file "
            "only if you are certain no deadline has passed."
            % (track, state["gameweek"], gameweek))

    value_column = TRACKS[track]
    maps = _maps(frame, value_column)
    squad = _squad_from_state(state, season)
    squad.award_free_transfer()

    plan = plan_transfers(squad, maps["projections"], maps["prices"], maps["clubs"],
                          maps["positions"], candidates=list(maps["projections"].keys()),
                          **POLICY)
    execute(squad, plan)
    return _state_from_squad(squad, season, gameweek, moves=plan.moves,
                             history=state.get("history", []))


def track_block(frame: pd.DataFrame, state: dict, track: str,
                season: str) -> dict:
    """The locked, scoreable record of a track's team for this gameweek.

    Shaped like the other variant blocks so `scorer.score_artifact` can score it with
    no special case — plus `hits`, which the scorer subtracts. A carried squad is the
    only thing here that can cost points to assemble.
    """
    value_column = TRACKS[track]
    maps = _maps(frame, value_column)
    squad = _squad_from_state(state, season)
    choice = best_xi(squad.players, maps["projections"], maps["positions"])
    if choice is None:
        return {"name": track, "error": "no legal XI"}

    moves = (state.get("history") or [{}])[-1].get("moves", [])
    ranked = sorted(choice.xi, key=lambda pid: -maps["projections"].get(pid, 0.0))
    names = dict(zip(frame["player_id"].astype(int), frame["player"]))
    return {
        "name": track,
        "value_column": value_column,
        "xi": [int(pid) for pid in choice.xi],
        "bench": [int(pid) for pid in choice.bench],
        "captain_id": int(choice.captain),
        "vice_captain_id": int(ranked[1]) if len(ranked) > 1 else int(choice.captain),
        # `projected` already counts the captain twice, as the gameweek is worth.
        "projected_points": round(float(choice.projected), 3),
        # POLICY forbids hits, so this is structurally 0 — recorded explicitly rather
        # than omitted, so the scorer's subtraction is exercised on every track and a
        # future policy that DOES take hits cannot slip through unscored.
        "hits": 0,
        "transfers": [{"out": int(m["out"]), "in": int(m["in"]),
                       "out_name": str(names.get(int(m["out"]), m["out"])),
                       "in_name": str(names.get(int(m["in"]), m["in"])),
                       "gain": round(float(m.get("gain", 0.0)), 3)} for m in moves],
        "bank": round(state["bank"] / float(TENTHS_PER_MILLION), 1),
        "free_transfers": int(state["free_transfers"]),
        "squad_value": round(
            squad.value_tenths(maps["prices"]) / float(TENTHS_PER_MILLION), 1),
    }


def advance_tracks(frame: pd.DataFrame, opening_squad_ids: List[int], season: str,
                   gameweek: int, persist: bool = True) -> dict:
    """Advance (or open) both tracks and return their locked blocks.

    Returns {track: block}. On the first gameweek both tracks open from
    `opening_squad_ids` and make no transfer, so they are identical by construction —
    which is correct: they have not yet decided anything to differ about.
    """
    blocks = {}
    for track in TRACKS:
        state = load_state(season, track)
        if state is None:
            state = open_track(frame, opening_squad_ids, season, gameweek, track)
        elif int(state["gameweek"]) < int(gameweek):
            state = step(frame, state, track, season, gameweek)
        else:
            logger.info("%s already decided GW%s; reusing the locked state",
                        track, state["gameweek"])
        if persist:
            save_state(season, track, state)
        blocks[track] = track_block(frame, state, track, season)
    return blocks
