"""Historical per-gameweek FPL data for seasons the live API no longer serves.

The live `element-summary` endpoint only exposes the CURRENT season, so a
one-season backtest is all it can support - and one season of 33 gameweeks has
too little power to separate a real edge from noise. The community archive
(github.com/vaastav/Fantasy-Premier-League) mirrors the same endpoints, scraped
weekly since 2016/17, and is the standard reference dataset for FPL research.

    Source:  https://github.com/vaastav/Fantasy-Premier-League
    Licence: Creative Commons Attribution 4.0 (CC BY 4.0)
    Files:   data/<season>/gws/merged_gw.csv   one row per player per fixture
             data/<season>/teams.csv           id -> team name for that season

WHY ONLY 2022/23 ONWARD, when the archive reaches back to 2016/17:

    `expected_goals` and `expected_assists` (Opta xG/xA) first appear in FPL's
    own data in 2022/23. Our projection is built on them. Replaying 2016/17 on
    *realised* goals and assists would not be a longer test of this model - it
    would be a test of a DIFFERENT model that we never benchmarked. Padding the
    sample by silently swapping the inputs is exactly the kind of thing this
    project exists not to do. Four seasons it is.

Two further season-to-season differences, both handled rather than ignored:

  - `defensive_contribution` is a 2025/26 rule. Earlier seasons have no such
    column and no such points, so the term is simply absent from both the
    projection and the actual points. Consistent within each season.
  - 2024/25 introduced manager elements (position "AM"). They are not players,
    have no xG, and our scoring rules do not cover them. They are dropped, and
    the count is logged rather than passed over in silence.

Snapshots are written into the same versioned, checksummed raw lake as every
other source, so a backtest never re-fetches and stays reproducible.
"""

import io
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
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
from research.data.fpl_loader import canonical_team

logger = logging.getLogger(__name__)

SOURCE_NAME = "fpl_archive"
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; football-prediction-engine/1.0)"}
REQUEST_TIMEOUT_SECONDS = 60

# Seasons in which FPL publishes expected_goals / expected_assists. See module docstring.
SEASONS_WITH_XG: Tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")

DATASETS = {"merged_gw": "gws/merged_gw.csv", "teams": "teams.csv"}

# FPL position labels -> element_type codes used throughout prediction_engine.fpl.
POSITION_CODES: Dict[str, int] = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
# Manager elements (2024/25+). Not players; excluded.
NON_PLAYER_POSITIONS = ("AM", "MNG")

PRICE_UNITS_PER_MILLION = 10.0

# Per-fixture fields the backtest consumes. `defensive_contribution` is optional.
NUMERIC_FIELDS = (
    "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves",
    "bonus", "yellow_cards", "red_cards", "own_goals", "penalties_missed",
    "penalties_saved", "total_points", "value", "opponent_team",
)
FLOAT_FIELDS = ("expected_goals", "expected_assists")
OPTIONAL_NUMERIC_FIELDS = ("defensive_contribution",)


_TRUE_STRINGS = {"true", "1", "t", "yes"}
_FALSE_STRINGS = {"false", "0", "f", "no"}


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def _dataset_dir(season: str, dataset: str):
    return load_config().raw_data_dir / SOURCE_NAME / season / dataset


def _ensure_dataset(season: str, dataset: str, force: bool = False) -> bytes:
    """Fetch one archive file once, into an immutable versioned snapshot."""
    dataset_dir = _dataset_dir(season, dataset)
    filename = f"{dataset}.csv"

    if not force and has_any_version(dataset_dir):
        version = read_latest_version(dataset_dir)
        return (dataset_dir / version / filename).read_bytes()

    url = f"{BASE_URL}/{season}/{DATASETS[dataset]}"
    logger.info("fetching %s", url)
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content

    version = new_version_id()
    metadata = build_metadata(
        source=SOURCE_NAME,
        identifier=f"{season}/{dataset}",
        source_url=url,
        version=version,
        local_path=dataset_dir / version / filename,
        content=content,
        checksum_sha256=sha256_bytes(content),
    )
    write_new_version(dataset_dir, filename, content, metadata)
    return content


def load_team_names(season: str, force: bool = False) -> Dict[int, str]:
    """FPL team id -> canonical (Understat) team name, for one season.

    Team ids are re-assigned alphabetically every season, so a table fetched for
    2023/24 will silently mislabel 2022/23. Always load the season's own.
    """
    frame = pd.read_csv(io.BytesIO(_ensure_dataset(season, "teams", force)))
    return {int(row["id"]): canonical_team(row["name"]) for _, row in frame.iterrows()}


def load_gameweeks(season: str, force: bool = False) -> pd.DataFrame:
    """One row per player per fixture, typed, with canonical team names.

    Columns: season, gameweek, player_id, player, position (int), team, opponent,
    was_home, kickoff_time, price, plus the per-fixture stat fields.
    """
    if season not in SEASONS_WITH_XG:
        raise ValueError(
            f"Season '{season}' has no expected_goals in FPL's data; "
            f"replaying it would test a different model. Available: {SEASONS_WITH_XG}"
        )

    raw = pd.read_csv(io.BytesIO(_ensure_dataset(season, "merged_gw", force)))
    teams = load_team_names(season, force)

    managers = raw["position"].isin(NON_PLAYER_POSITIONS).sum()
    if managers:
        logger.info("%s: dropping %d manager rows (not players)", season, managers)
    raw = raw[~raw["position"].isin(NON_PLAYER_POSITIONS)]

    unknown = set(raw["position"]) - set(POSITION_CODES)
    if unknown:
        raise ValueError(f"{season}: unrecognised positions {sorted(unknown)}")

    frame = pd.DataFrame({
        "season": season,
        "gameweek": raw["GW"].astype(int),
        "player_id": raw["element"].astype(int),
        "player": raw["name"].astype(str),
        "position": raw["position"].map(POSITION_CODES).astype(int),
        "team": raw["team"].astype(str).map(canonical_team),
        # Never `astype(bool)`: on the string "False" that silently yields True,
        # which would mark every fixture a home fixture.
        "was_home": raw["was_home"].map(_as_bool),
        "kickoff_time": pd.to_datetime(raw["kickoff_time"], utc=True),
    })
    for field in NUMERIC_FIELDS:
        frame[field] = pd.to_numeric(raw[field], errors="coerce").fillna(0).astype(int)
    for field in FLOAT_FIELDS:
        frame[field] = pd.to_numeric(raw[field], errors="coerce").fillna(0.0).astype(float)
    for field in OPTIONAL_NUMERIC_FIELDS:
        frame[field] = (
            pd.to_numeric(raw[field], errors="coerce").fillna(0).astype(int)
            if field in raw.columns else 0
        )

    frame["opponent"] = frame["opponent_team"].map(teams)
    frame["price"] = frame["value"] / PRICE_UNITS_PER_MILLION

    missing = frame["opponent"].isna().sum()
    if missing:
        raise ValueError(f"{season}: {missing} rows have an unmappable opponent id")

    logger.info(
        "%s: %d player-fixtures, %d gameweeks, %d players",
        season, len(frame), frame["gameweek"].nunique(), frame["player_id"].nunique(),
    )
    return frame.drop(columns=["opponent_team", "value"])


def unmapped_teams(season: str, engine_teams) -> list:
    """Archive teams whose canonical name the prediction engine does not know.
    Empty means the join is complete; anything else would silently drop a club."""
    known = set(engine_teams)
    return sorted(t for t in load_team_names(season).values() if t not in known)
