"""Joins API-Football injury data onto the Understat match dataset.

Turns the raw per-fixture injury records (ingested under
data/raw/api_football/injuries/) into a simple per-match availability feature:
how many players each side is MISSING for a given match. That is the point-in-
time "who is unavailable" signal Phase 3b tests.

Two sources, so a small join is needed. It is deliberately narrow and
validated:
- Team names: API-Football and Understat agree on 21 of 24 clubs; the three
  that differ (Newcastle / Sheffield Utd / Wolves) are mapped explicitly.
- Match key: (calendar date, team). A team plays at most once per day, so this
  is unique. API-Football timestamps are UTC and tz-aware; they are converted
  to tz-naive dates to line up with Understat's naive datetimes.

Only the three seasons the free API plan grants (2022/23-2024/25) carry injury
data; for any other season the injury features are left NaN (unknown), so a
model trained/evaluated outside the covered window is never fed a fake zero.
Within the covered window, a match-team with no injury record genuinely had
zero players missing, so it is filled with 0.

We count only players flagged "Missing Fixture" (a definite absence), not
"Questionable" (a doubt), as the availability signal.
"""

import json
import logging

import numpy as np
import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version
from research.data.player_importance import build_importance_lookup

logger = logging.getLogger(__name__)

SOURCE_NAME = "api_football"
INJURY_TYPE_OUT = "Missing Fixture"

# API-Football team name -> Understat canonical name (only the 3 that differ).
TEAM_NAME_MAP = {
    "Newcastle": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
    "Wolves": "Wolverhampton Wanderers",
}

# Understat season label -> API-Football season code (start year). These three
# seasons are the free plan's injury coverage window.
COVERED_SEASONS = {"2022-23": "2022", "2023-24": "2023", "2024-25": "2024"}

INJURY_FEATURES = ["home_injuries", "away_injuries", "injuries_diff"]

# Importance-weighted absences: each missing player contributes his share of a
# full season's minutes LAST season, so a missing regular counts ~1.0 and a
# missing fringe player counts ~0.05. Phase 3c's answer to the blunt raw count.
INJURY_WEIGHT_FEATURES = [
    "home_injury_weight",
    "away_injury_weight",
    "injury_weight_diff",
]


def _canonical(af_team_name: str) -> str:
    return TEAM_NAME_MAP.get(af_team_name, af_team_name)


def _load_season_records(raw_data_dir, league: str, af_season: str) -> list:
    dataset_dir = raw_data_dir / SOURCE_NAME / "injuries" / league / af_season
    version = read_latest_version(dataset_dir)
    if version is None:
        raise ValueError(
            f"No ingested injuries for league {league} season {af_season} - run "
            f"the ingest CLI first (needs APIFOOTBALL_KEY)."
        )
    path = dataset_dir / version / f"injuries_{league}_{af_season}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _previous_af_season(af_season: str) -> str:
    return str(int(af_season) - 1)


def load_missing_players(league: str = "39") -> pd.DataFrame:
    """Return one row per missing player: (date, team, player, importance).

    `importance` is the player's share of a full season of minutes in the
    PREVIOUS season (leakage-safe; see research/data/player_importance.py).
    """
    config = load_config()
    raw = config.raw_data_dir

    rows = []
    for season_label, af_season in COVERED_SEASONS.items():
        # Importance comes from the season BEFORE the one being played.
        lookup = build_importance_lookup(_previous_af_season(af_season))
        for rec in _load_season_records(raw, league, af_season):
            if rec["player"]["type"] != INJURY_TYPE_OUT:
                continue
            player_name = rec["player"]["name"]
            rows.append(
                {
                    "date": pd.to_datetime(rec["fixture"]["date"]).tz_convert(None).normalize(),
                    "team": _canonical(rec["team"]["name"]),
                    "season": season_label,
                    "player": player_name,
                    "importance": lookup.importance(player_name),
                }
            )
    return pd.DataFrame(rows)


def load_missing_counts(league: str = "39") -> pd.DataFrame:
    """Per (date, team): how many players were missing, and their summed
    importance (the weighted version of the same absence list)."""
    players = load_missing_players(league)
    grouped = players.groupby(["date", "team"]).agg(
        missing=("player", "size"), weight=("importance", "sum")
    )
    return grouped.reset_index()


def add_injury_features(matches: pd.DataFrame, league: str = "39") -> pd.DataFrame:
    """Add both the raw-count and the importance-weighted injury features.

    Covered seasons get real values (0 where a team had nobody missing); all
    other seasons get NaN so a model never mistakes 'no data' for 'nobody hurt'.
    """
    counts = load_missing_counts(league)
    count_lookup = {(r.date, r.team): int(r.missing) for r in counts.itertuples()}
    weight_lookup = {(r.date, r.team): float(r.weight) for r in counts.itertuples()}

    home_n, away_n, home_w, away_w = [], [], [], []
    for row in matches.itertuples():
        if row.season in COVERED_SEASONS:
            day = pd.Timestamp(row.date).normalize()
            home_n.append(float(count_lookup.get((day, row.home_team), 0.0)))
            away_n.append(float(count_lookup.get((day, row.away_team), 0.0)))
            home_w.append(weight_lookup.get((day, row.home_team), 0.0))
            away_w.append(weight_lookup.get((day, row.away_team), 0.0))
        else:
            home_n.append(np.nan)
            away_n.append(np.nan)
            home_w.append(np.nan)
            away_w.append(np.nan)

    df = matches.copy()
    df["home_injuries"] = home_n
    df["away_injuries"] = away_n
    df["injuries_diff"] = df["home_injuries"] - df["away_injuries"]
    df["home_injury_weight"] = home_w
    df["away_injury_weight"] = away_w
    df["injury_weight_diff"] = df["home_injury_weight"] - df["away_injury_weight"]

    covered = df["season"].isin(COVERED_SEASONS)
    logger.info(
        "Injury features attached to %d matches (mean per side - count: "
        "home=%.2f away=%.2f | weight: home=%.2f away=%.2f)",
        int(covered.sum()),
        df.loc[covered, "home_injuries"].mean(),
        df.loc[covered, "away_injuries"].mean(),
        df.loc[covered, "home_injury_weight"].mean(),
        df.loc[covered, "away_injury_weight"].mean(),
    )
    return df
