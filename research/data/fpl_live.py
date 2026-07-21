"""Score a gameweek from FPL's own live endpoint, the day it finishes.

`scorer.gameweek_actuals` reads a community archive (vaastav's repo). That is right
for the eight-season backtest — it is complete and it is what the proof was computed
on — but in-season it lags the live game by a few days. The Bank-It ledger wants to
score each gameweek as soon as it is done, so it reads FPL directly:

    GET /api/event/{gw}/live/  ->  elements[].stats.{total_points, minutes}

The `stats` block is already the player's TOTAL for the gameweek across all his
fixtures, so a double gameweek is summed by FPL, not by us — matching how the locked
squad's additive channels were summed at projection time. Keyed by the current
season's element `id`, which is exactly the id the locked artifact stores, so no
cross-season code join is needed (that hazard only exists between seasons).

INTEGRITY — the one thing this module must get right. During matches, and for a while
after, `total_points` carries PROVISIONAL bonus; it is not final until FPL confirms
the bonus and sets the event's `data_checked` flag. Scoring a gameweek before then
would silently record numbers that later change — the exact hindsight-adjacent error
the whole pipeline is built to avoid, just arriving from the other direction. So
`live_actuals` refuses to score a gameweek that is not `data_checked`, unless the
caller explicitly opts into a provisional read and is told so.

stdlib + requests only, no pandas and no ingested lake, so it runs anywhere the live
loop runs — the same standalone discipline as `predicted_lineups` and
`fpl_season_watch`.
"""

import json
import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
LIVE_URL = "https://fantasy.premierleague.com/api/event/{gameweek}/live/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
TIMEOUT = 30


class LiveScoringError(RuntimeError):
    """Raised when a gameweek cannot be scored safely (not final, or not published)."""


def _get(url: str) -> dict:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_bootstrap() -> dict:
    return _get(BOOTSTRAP_URL)


def fetch_live(gameweek: int) -> dict:
    return _get(LIVE_URL.format(gameweek=int(gameweek)))


def gameweek_finality(bootstrap: dict, gameweek: int) -> dict:
    """What FPL says about a gameweek's completeness. Returns the two flags that
    matter and a `scoreable` verdict derived from them.

    `finished`      every match has kicked off and basic points are in.
    `data_checked`  FPL has CONFIRMED the gameweek, including bonus. This is the only
                    safe signal to score on: before it, bonus is provisional.
    """
    event = next((e for e in (bootstrap.get("events") or [])
                  if int(e.get("id", -1)) == int(gameweek)), None)
    if event is None:
        return {"exists": False, "finished": False, "data_checked": False,
                "scoreable": False}
    finished = bool(event.get("finished"))
    data_checked = bool(event.get("data_checked"))
    return {"exists": True, "finished": finished, "data_checked": data_checked,
            "scoreable": data_checked}


def parse_live(payload: dict) -> Dict[int, dict]:
    """{element_id: {"points": p, "minutes": m}} from an event/{gw}/live payload.

    Exactly the shape of `scorer.gameweek_actuals`, so the ledger consumes either
    source without a special case. `minutes` is carried because the vice-captain rule
    needs it (a captain who played 0 is replaced).
    """
    actuals = {}
    for element in (payload.get("elements") or []):
        stats = element.get("stats") or {}
        actuals[int(element["id"])] = {
            "points": float(stats.get("total_points", 0.0)),
            "minutes": float(stats.get("minutes", 0.0)),
        }
    return actuals


def live_actuals(gameweek: int, require_final: bool = True,
                 bootstrap: Optional[dict] = None) -> Dict[int, dict]:
    """Actual points and minutes for a completed gameweek, from FPL live.

    Refuses by default to score a gameweek FPL has not `data_checked`, because its
    bonus is still provisional and the numbers will move. Pass `require_final=False`
    only for an explicitly-labelled provisional read (e.g. a within-minutes preview),
    never for the ledger.
    """
    bootstrap = bootstrap if bootstrap is not None else fetch_bootstrap()
    finality = gameweek_finality(bootstrap, gameweek)
    if not finality["exists"]:
        raise LiveScoringError("GW%d is not published in the bootstrap" % gameweek)
    if require_final and not finality["scoreable"]:
        raise LiveScoringError(
            "GW%d is not final yet (finished=%s, data_checked=%s); its bonus is "
            "provisional. Refusing to score it into the ledger."
            % (gameweek, finality["finished"], finality["data_checked"]))
    if not require_final and not finality["scoreable"]:
        logger.warning("GW%d is not data_checked; returning PROVISIONAL points that "
                       "may still change (bonus not confirmed)", gameweek)
    return parse_live(fetch_live(gameweek))
