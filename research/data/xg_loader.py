"""Builds a clean match + Expected Goals (xG) research dataset from the
Understat raw lake.

Like `loader.py` (which reads football-data.co.uk), this is a research-grade,
read-only transform of raw snapshots into one clean pandas DataFrame - it does
not fetch anything itself. It reads whatever the ingest layer has stored under
`data/raw/understat/<league>/<season>/`.

Understat's payload is self-contained: each match carries both teams, actual
goals, AND xG, so no cross-source join (with its team-name-matching hazards)
is needed - the H/D/A result is derived from Understat's own goals. That makes
this dataset an internally-consistent basis for a fair comparison between
goal-based and xG-based models on an identical set of matches.

Output columns (one row per played match, chronologically sorted):
    date, league, season, home_team, away_team, home_goals, away_goals,
    result (H/D/A), home_xg, away_xg
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version

logger = logging.getLogger(__name__)

SOURCE_NAME = "understat"


def _season_label(start_year: str) -> str:
    """'2014' -> '2014-15' (the calendar year a season started -> its label).
    Sortable lexicographically, so it can be used directly for walk-forward
    ordering."""
    end = str(int(start_year) + 1)[-2:]
    return f"{start_year}-{end}"


def _latest_json_path(
    raw_data_dir: Path, league: str, start_year: str
) -> Optional[Path]:
    dataset_dir = raw_data_dir / SOURCE_NAME / league / start_year
    version = read_latest_version(dataset_dir)
    if version is None:
        return None
    return dataset_dir / version / f"{league}_{start_year}.json"


def _load_season(json_path: Path, league: str, start_year: str) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    rows = []
    for match in payload["dates"]:
        # Skip fixtures not yet played (no result / no xG). For a completed
        # historical season every match has isResult=True, but guarding keeps
        # the loader correct if an in-progress season is ever ingested.
        if not match.get("isResult"):
            continue
        goals = match["goals"]
        xg = match["xG"]
        home_goals = int(goals["h"])
        away_goals = int(goals["a"])
        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"
        rows.append(
            {
                "date": pd.to_datetime(match["datetime"]),
                "league": league,
                "season": _season_label(start_year),
                "home_team": match["h"]["title"].strip(),
                "away_team": match["a"]["title"].strip(),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
                "home_xg": float(xg["h"]),
                "away_xg": float(xg["a"]),
            }
        )

    if not rows:
        raise ValueError(f"{json_path}: no played matches found in payload")

    return pd.DataFrame(rows)


def load_understat_matches(
    league: str = "EPL", seasons: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load every available Understat season for `league` into one clean,
    chronologically-sorted DataFrame.

    `league` is Understat's own code (e.g. "EPL"). `seasons` (start-year codes
    like "2023") restricts to a subset; omit to load every season configured
    for Understat in config.yaml that has actually been downloaded.
    """
    config = load_config()
    raw_data_dir = config.raw_data_dir
    all_seasons = seasons if seasons is not None else list(config.understat.seasons)

    frames = []
    for start_year in all_seasons:
        json_path = _latest_json_path(raw_data_dir, league, start_year)
        if json_path is None:
            logger.warning(
                "No downloaded version found for understat %s/%s - skipping "
                "(has it been fetched via the ingest CLI yet?)",
                league,
                start_year,
            )
            continue
        frames.append(_load_season(json_path, league, start_year))

    if not frames:
        raise ValueError(
            f"No Understat data available for league '{league}' - run the "
            f"ingest downloader first (python -m data_warehouse.cli download "
            f"--source understat --leagues {league})."
        )

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    return matches
