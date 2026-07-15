"""Per-match xG/xA for every Understat player, cached in the versioned raw lake.

FPL only began publishing per-player xG in 2022/23. To backtest the FPL projection
on earlier seasons *without* silently changing the model, the xG has to come from
somewhere that measured it at the time. Understat did, back to 2014/15.

The league feed we already ingest carries season TOTALS per player, which cannot
drive a walk-forward backtest (projecting gameweek k may use only 1..k-1). This
fetches the per-MATCH log instead, from the endpoint the player pages call:

    GET understat.com/main/getPlayerMatches/{id}   ->  {response:{matches:[...]}}

One request returns a player's whole career, so it is fetched once per player, not
per season. Only EPL-relevant fields are kept; a player's matches in other leagues
(a mid-career transfer abroad) are harmless - they simply never join to an FPL
gameweek - but they are dropped to keep the snapshot small.

Like `fpl_histories`, this writes one immutable, checksummed snapshot into the lake
and never touches the network again once cached, so the backtest stays reproducible.
"""

import glob
import json
import logging
import os
import time
from typing import Dict, List

import requests

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import (
    build_metadata,
    has_any_version,
    new_version_id,
    read_latest_version,
    write_new_version,
)
from data_warehouse.utils.checksum import sha256_bytes

logger = logging.getLogger(__name__)

ENDPOINT = "https://understat.com/main/getPlayerMatches/{id}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "X-Requested-With": "XMLHttpRequest",   # the feed 404s without it
}
REQUEST_DELAY_SECONDS = 0.3
DATASET = "player-matches"
FILENAME = "player-matches.json"

# The seasons we need player-level xG for. Understat labels a season by the calendar
# year it starts, exactly as the league feed does.
FIRST_SEASON = 2015
LAST_SEASON = 2025

# The per-match fields a projection or a scoring reconstruction needs.
# npg/npxG are non-penalty goals / xG: penalty xG = xG - npxG per match, the
# leakage-safe penalty signal (Phase 6c). A player only accrues penalty xG when
# he actually takes the kick, so this doubles as the taker identifier.
KEEP_FIELDS = ("season", "date", "xG", "xA", "goals", "assists", "time",
               "h_team", "a_team", "h_goals", "a_goals", "npg", "npxG")

UNDERSTAT_SOURCE = "understat"
LEAGUE = "EPL"


def _dataset_dir():
    return load_config().raw_data_dir / UNDERSTAT_SOURCE / LEAGUE / DATASET


def _league_season_dir(start_year: int):
    return load_config().raw_data_dir / UNDERSTAT_SOURCE / LEAGUE / str(start_year)


def _load_league_season(start_year: int) -> dict:
    directory = _league_season_dir(start_year)
    version = read_latest_version(directory)
    if version is None:
        raise ValueError("No ingested Understat %s %s; the league feed must be "
                         "downloaded first" % (LEAGUE, start_year))
    path = [p for p in glob.glob(os.path.join(str(directory), version, "*.json"))
            if not p.endswith(".meta.json")][0]
    return json.loads(open(path, encoding="utf-8").read())


def player_universe(first: int = FIRST_SEASON, last: int = LAST_SEASON) -> Dict[str, str]:
    """Every Understat player id that appeared in the EPL across the seasons, with a
    name (the most recent one seen). This is the join key set and the fetch list."""
    names = {}
    for start_year in range(first, last + 1):
        try:
            payload = _load_league_season(start_year)
        except ValueError:
            logger.warning("no league feed for %d; skipping", start_year)
            continue
        for player in payload["players"]:
            names[player["id"]] = player["player_name"]
    return names


def _fetch_one(player_id: str) -> List[dict]:
    response = requests.get(ENDPOINT.format(id=player_id), headers=HEADERS, timeout=30)
    response.raise_for_status()
    matches = response.json()["response"]["matches"]
    kept = []
    for match in matches:
        season = int(match.get("season", -1))
        if FIRST_SEASON <= season <= LAST_SEASON:
            kept.append({field: match.get(field) for field in KEEP_FIELDS})
    return kept


def _fetch_all(player_ids: List[str]) -> Dict[str, list]:
    matches = {}
    for i, player_id in enumerate(player_ids, start=1):
        try:
            matches[player_id] = _fetch_one(player_id)
        except Exception as error:               # a single dead id must not lose the run
            logger.warning("player %s failed: %s", player_id, error)
            matches[player_id] = []
        if i % 200 == 0:
            logger.info("fetched %d/%d player match logs", i, len(player_ids))
        time.sleep(REQUEST_DELAY_SECONDS)
    return matches


def ensure_player_matches(force: bool = False) -> Dict[str, list]:
    """Return every player's per-match xG log, fetching once and caching."""
    dataset_dir = _dataset_dir()

    if not force and has_any_version(dataset_dir):
        version = read_latest_version(dataset_dir)
        path = dataset_dir / version / FILENAME
        logger.info("using cached Understat player matches (%s)", version)
        return json.loads(path.read_text(encoding="utf-8"))

    universe = player_universe()
    logger.info("fetching per-match xG for %d Understat players", len(universe))
    matches = _fetch_all(sorted(universe))

    content = json.dumps(matches).encode("utf-8")
    version = new_version_id()
    metadata = build_metadata(
        source=UNDERSTAT_SOURCE,
        identifier="%s/%s" % (LEAGUE, DATASET),
        source_url=ENDPOINT.format(id="{id}"),
        version=version,
        local_path=dataset_dir / version / FILENAME,
        content=content,
        checksum_sha256=sha256_bytes(content),
    )
    write_new_version(dataset_dir, FILENAME, content, metadata)
    total = sum(len(v) for v in matches.values())
    logger.info("cached %d players, %d matches (%d bytes)",
                len(matches), total, len(content))
    return matches
