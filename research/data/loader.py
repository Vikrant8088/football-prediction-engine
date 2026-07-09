"""Builds a clean research dataset from the versioned raw data lake.

This reads whatever `data_warehouse`'s ingest layer (Subsystem 1) has already
downloaded from football-data.co.uk under `data/raw/football_data_co_uk/`.
It does not fetch anything itself and does not touch the canonical/domain
model design (Subsystem 2) - this is a research-grade, read-only transform
of raw CSV snapshots into a single clean pandas DataFrame, scoped to exactly
what the benchmark experiments need: match date, season, teams, and goals.

football-data.co.uk's column set has changed over the years (extra odds
columns, a "Time" column added in recent seasons, a UTF-8 BOM on newer
files) - this loader only ever reads the small set of core columns that has
stayed stable since the site's earliest seasons, and ignores everything else.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version

logger = logging.getLogger(__name__)

SOURCE_NAME = "football_data_co_uk"

# The only columns this loader depends on. Present, under these exact names,
# in every football-data.co.uk season file back to the early 2000s.
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def _season_label(season_code: str) -> str:
    """'0910' -> '2009-10'. Assumes 21st-century seasons (fine for the
    2000s-2020s range this research dataset covers)."""
    return f"20{season_code[:2]}-{season_code[2:]}"


def _dataset_dir(raw_data_dir: Path, league_code: str, season_code: str) -> Path:
    return raw_data_dir / SOURCE_NAME / league_code / season_code


def _latest_csv_path(raw_data_dir: Path, league_code: str, season_code: str) -> Optional[Path]:
    dataset_dir = _dataset_dir(raw_data_dir, league_code, season_code)
    version = read_latest_version(dataset_dir)
    if version is None:
        return None
    return dataset_dir / version / f"{season_code}.csv"


def _load_season(csv_path: Path, league_code: str, season_code: str) -> pd.DataFrame:
    # utf-8-sig transparently strips the BOM that recent seasons' files carry.
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    missing = [c for c in CORE_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected column(s) {missing}; "
            f"found columns: {list(raw.columns)}"
        )

    df = raw[CORE_COLUMNS].copy()
    df = df.dropna(subset=CORE_COLUMNS)

    # Season files mix 2-digit years (older seasons) and 4-digit years (
    # recent seasons); dayfirst=True handles both, and pandas infers the
    # per-file format once from the first parseable value.
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["league"] = league_code
    df["season"] = _season_label(season_code)
    df["home_team"] = df["HomeTeam"].str.strip()
    df["away_team"] = df["AwayTeam"].str.strip()
    df["home_goals"] = df["FTHG"].astype(int)
    df["away_goals"] = df["FTAG"].astype(int)
    df["result"] = df["FTR"].str.strip()

    computed_result = pd.Series("D", index=df.index)
    computed_result[df["home_goals"] > df["away_goals"]] = "H"
    computed_result[df["home_goals"] < df["away_goals"]] = "A"
    mismatches = df["result"] != computed_result
    if mismatches.any():
        raise ValueError(
            f"{csv_path}: {mismatches.sum()} row(s) where FTR disagrees with "
            f"FTHG/FTAG - data integrity issue, refusing to silently pick one"
        )

    return df[
        ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals", "result"]
    ]


def load_league_matches(league_code: str, seasons: Optional[List[str]] = None) -> pd.DataFrame:
    """Load every available season for `league_code` into one clean,
    chronologically-sorted DataFrame.

    `seasons` (raw season codes like "2324") restricts to a subset; omit to
    load every season configured for football-data.co.uk in config.yaml that
    has actually been downloaded into the raw lake.
    """
    config = load_config()
    raw_data_dir = config.raw_data_dir
    all_seasons = seasons if seasons is not None else list(config.football_data_co_uk.seasons)

    frames = []
    for season_code in all_seasons:
        csv_path = _latest_csv_path(raw_data_dir, league_code, season_code)
        if csv_path is None:
            logger.warning(
                "No downloaded version found for %s/%s - skipping (has it been "
                "fetched via the ingest CLI yet?)",
                league_code,
                season_code,
            )
            continue
        frames.append(_load_season(csv_path, league_code, season_code))

    if not frames:
        raise ValueError(
            f"No data available for league '{league_code}' - run the ingest "
            f"downloader first."
        )

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    return matches
