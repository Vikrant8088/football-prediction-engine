"""Per-gameweek FPL history for every player, cached in the versioned raw lake.

`bootstrap-static` gives season TOTALS, which are useless for a walk-forward
backtest: to project gameweek k+1 honestly you may only use gameweeks 1..k. The
per-player `element-summary` endpoint gives exactly that, one request per player.

841 requests is too many for the warehouse's shared 1-request-per-second policy,
so this fetches with its own polite delay and writes one combined, immutable,
checksummed snapshot into the lake. Once cached, the backtest never touches the
network again - so it stays reproducible.

Only the fields a projection or a scoring reconstruction needs are kept; the raw
payload also carries transfer counts, ICT indices and prices we do not use, and
storing them would triple the file for nothing.
"""

import json
import logging
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
from research.data.fpl_loader import load_players

logger = logging.getLogger(__name__)

API = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
REQUEST_DELAY_SECONDS = 0.35
DATASET = "element-histories"
FILENAME = "element-histories.json"

# The per-match fields we actually use.
KEEP_FIELDS = (
    "round", "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "bonus", "yellow_cards", "red_cards", "own_goals", "penalties_missed",
    "penalties_saved", "defensive_contribution", "expected_goals", "expected_assists",
    "total_points", "opponent_team", "was_home", "kickoff_time",
)


def _dataset_dir():
    return load_config().raw_data_dir / "fpl" / DATASET


def _fetch_all(player_ids: List[int]) -> Dict[str, list]:
    histories = {}
    for i, player_id in enumerate(player_ids, start=1):
        response = requests.get(f"{API}/element-summary/{player_id}/", headers=HEADERS, timeout=30)
        response.raise_for_status()
        rows = response.json()["history"]
        histories[str(player_id)] = [
            {field: row.get(field) for field in KEEP_FIELDS} for row in rows
        ]
        if i % 100 == 0:
            logger.info("fetched %d/%d player histories", i, len(player_ids))
        time.sleep(REQUEST_DELAY_SECONDS)
    return histories


def ensure_histories(force: bool = False) -> Dict[str, list]:
    """Return every player's per-gameweek history, fetching once and caching."""
    dataset_dir = _dataset_dir()

    if not force and has_any_version(dataset_dir):
        version = read_latest_version(dataset_dir)
        path = dataset_dir / version / FILENAME
        logger.info("using cached FPL histories (%s)", version)
        return json.loads(path.read_text(encoding="utf-8"))

    player_ids = load_players()["id"].tolist()
    logger.info("fetching per-gameweek history for %d players", len(player_ids))
    histories = _fetch_all(player_ids)

    content = json.dumps(histories).encode("utf-8")
    version = new_version_id()
    metadata = build_metadata(
        source="fpl",
        identifier=DATASET,
        source_url=f"{API}/element-summary/{{id}}/",
        version=version,
        local_path=dataset_dir / version / FILENAME,
        content=content,
        checksum_sha256=sha256_bytes(content),
    )
    write_new_version(dataset_dir, FILENAME, content, metadata)
    logger.info("cached %d player histories (%d bytes)", len(histories), len(content))
    return histories
