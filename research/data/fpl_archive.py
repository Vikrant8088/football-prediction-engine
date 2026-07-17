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
from research.data.csv_utils import read_csv_bytes_resilient
from research.data.fpl_loader import canonical_team

logger = logging.getLogger(__name__)

SOURCE_NAME = "fpl_archive"
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; football-prediction-engine/1.0)"}
REQUEST_TIMEOUT_SECONDS = 60

# Seasons in which FPL publishes expected_goals / expected_assists. Earlier seasons
# get their xG from Understat (`understat_xg_join`), so the MODEL is identical across
# all of them - the whole point of sourcing xG externally rather than dropping it.
SEASONS_WITH_XG: Tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")

# Every season the archive can be parsed for. 2016/17-2017/18 use an older format
# with no element/GW/kickoff columns and are excluded; 2018/19-2019/20 lack the
# per-row position/team columns, which are recovered from players_raw.csv.
ALL_SEASONS: Tuple[str, ...] = (
    "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
)

DATASETS = {
    "merged_gw": "gws/merged_gw.csv",
    "teams": "teams.csv",
    "players_raw": "players_raw.csv",   # element -> position/team, for seasons lacking them
}
# 2018/19 has no per-season teams.csv; this repo-root file maps (season, team id) ->
# name for every early season. One fetch, cached, shared across seasons.
MASTER_TEAM_LIST_URL = f"{BASE_URL}/master_team_list.csv"
MASTER_TEAM_DATASET = "master_team_list"

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


def _ensure_master_team_list(force: bool = False) -> bytes:
    dataset_dir = _dataset_dir("_shared", MASTER_TEAM_DATASET)
    filename = f"{MASTER_TEAM_DATASET}.csv"
    if not force and has_any_version(dataset_dir):
        version = read_latest_version(dataset_dir)
        return (dataset_dir / version / filename).read_bytes()
    logger.info("fetching %s", MASTER_TEAM_LIST_URL)
    response = requests.get(MASTER_TEAM_LIST_URL, headers=HEADERS,
                            timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    content = response.content
    version = new_version_id()
    metadata = build_metadata(
        source=SOURCE_NAME, identifier=MASTER_TEAM_DATASET, source_url=MASTER_TEAM_LIST_URL,
        version=version, local_path=dataset_dir / version / filename,
        content=content, checksum_sha256=sha256_bytes(content),
    )
    write_new_version(dataset_dir, filename, content, metadata)
    return content


def load_team_names(season: str, force: bool = False) -> Dict[int, str]:
    """FPL team id -> canonical (Understat) team name, for one season.

    Team ids are re-assigned alphabetically every season, so a table fetched for
    2023/24 will silently mislabel 2022/23. Always load the season's own. Seasons
    without a per-season teams.csv (2018/19) fall back to the repo-root
    master_team_list.csv, keyed by season.
    """
    try:
        frame = read_csv_bytes_resilient(_ensure_dataset(season, "teams", force),
                                         label=f"{season}/teams")
        return {int(row["id"]): canonical_team(row["name"])
                for _, row in frame.iterrows()}
    except requests.exceptions.HTTPError:
        master = read_csv_bytes_resilient(_ensure_master_team_list(force),
                                          label="master_team_list")
        rows = master[master["season"] == season]
        if rows.empty:
            raise ValueError(f"No team list for season '{season}'")
        return {int(r["team"]): canonical_team(r["team_name"]) for _, r in rows.iterrows()}


def load_player_meta(season: str, force: bool = False) -> Dict[int, dict]:
    """element id -> {position, team, name, code}, from players_raw.csv.

    Needed because 2018/19 and 2019/20 `merged_gw` rows carry neither position nor
    team; both are looked up here by the `element` id, which IS present.

    `code` is FPL's PERMANENT player code, stable across seasons — unlike the element
    id, which is reassigned every summer. It is the only safe key for joining one
    season's history onto another's squad; an id join would silently hand one player's
    record to whoever inherited his number.
    """
    frame = read_csv_bytes_resilient(_ensure_dataset(season, "players_raw", force),
                                     label=f"{season}/players_raw")
    teams = load_team_names(season, force)
    has_code = "code" in frame.columns
    meta = {}
    for _, row in frame.iterrows():
        code = None
        if has_code and pd.notna(row["code"]):
            code = int(row["code"])
        meta[int(row["id"])] = {
            "position": int(row["element_type"]),
            "team": teams.get(int(row["team"])),
            "name": ("%s %s" % (row["first_name"], row["second_name"])).strip(),
            "code": code,
        }
    return meta


def _clean_name(raw_name: str) -> str:
    """'Aaron_Cresswell_376' -> 'Aaron Cresswell'. Pre-2020 names carry underscores
    and a trailing element id; later seasons are already clean."""
    text = str(raw_name)
    if "_" in text:
        parts = [p for p in text.split("_") if not p.isdigit()]
        text = " ".join(parts)
    return text.strip()


def load_gameweeks(season: str, force: bool = False) -> pd.DataFrame:
    """One row per player per fixture, typed, with canonical team names.

    Columns: season, gameweek, player_id, player, position (int), team, opponent,
    was_home, kickoff_time, price, plus the per-fixture stat fields.
    """
    if season not in ALL_SEASONS:
        raise ValueError(
            f"Season '{season}' cannot be parsed. Available: {ALL_SEASONS}"
        )

    raw = read_csv_bytes_resilient(_ensure_dataset(season, "merged_gw", force),
                                   label=f"{season}/merged_gw")
    teams = load_team_names(season, force)

    # Manager elements exist only from 2024/25 and only when the column is present.
    if "position" in raw.columns:
        managers = raw["position"].isin(NON_PLAYER_POSITIONS).sum()
        if managers:
            logger.info("%s: dropping %d manager rows (not players)", season, managers)
        raw = raw[~raw["position"].isin(NON_PLAYER_POSITIONS)]

    # Position and team are per-row from 2020/21; earlier they come from players_raw.
    if "position" in raw.columns and raw["position"].notna().all():
        unknown = set(raw["position"]) - set(POSITION_CODES)
        if unknown:
            raise ValueError(f"{season}: unrecognised positions {sorted(unknown)}")
        position = raw["position"].map(POSITION_CODES).astype(int)
        team = raw["team"].astype(str).map(canonical_team)
    else:
        meta = load_player_meta(season, force)
        position = raw["element"].map(lambda e: (meta.get(int(e)) or {}).get("position"))
        team = raw["element"].map(lambda e: (meta.get(int(e)) or {}).get("team"))
        if position.isna().any():
            raise ValueError(f"{season}: {int(position.isna().sum())} rows have no "
                             f"players_raw position")

    frame = pd.DataFrame({
        "season": season,
        "gameweek": raw["GW"].astype(int),
        "player_id": raw["element"].astype(int),
        "player": raw["name"].map(_clean_name),
        "position": position.astype(int),
        "team": team,
        # Never `astype(bool)`: on the string "False" that silently yields True,
        # which would mark every fixture a home fixture.
        "was_home": raw["was_home"].map(_as_bool),
        "kickoff_time": pd.to_datetime(raw["kickoff_time"], utc=True),
    })
    for field in NUMERIC_FIELDS:
        frame[field] = pd.to_numeric(raw[field], errors="coerce").fillna(0).astype(int)
    # xG is present only from 2022/23; earlier seasons carry 0 here and have it
    # injected from Understat downstream. Do not confuse "0" with "no data".
    for field in FLOAT_FIELDS:
        frame[field] = (
            pd.to_numeric(raw[field], errors="coerce").fillna(0.0).astype(float)
            if field in raw.columns else 0.0
        )
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
        "%s: %d player-fixtures, %d gameweeks, %d players, xG from FPL: %s",
        season, len(frame), frame["gameweek"].nunique(), frame["player_id"].nunique(),
        season in SEASONS_WITH_XG,
    )
    return frame.drop(columns=["opponent_team", "value"])


def unmapped_teams(season: str, engine_teams) -> list:
    """Archive teams whose canonical name the prediction engine does not know.
    Empty means the join is complete; anything else would silently drop a club."""
    known = set(engine_teams)
    return sorted(t for t in load_team_names(season).values() if t not in known)
