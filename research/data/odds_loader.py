"""Closing bookmaker odds -> implied probabilities. The yardstick, never an input.

The betting market is the strongest publicly available forecast of football
matches: it aggregates everyone's information and money. The project's vision
lists "copy bookmaker predictions" as an explicit NON-GOAL, so odds are never
fed into a model. They are used for exactly one thing - to measure how good the
engine really is. If the engine cannot beat the closing line, real predictive
work remains; if it can, it is genuinely world-class.

Which odds: **Pinnacle closing** (`PSCH`/`PSCD`/`PSCA`), already sitting in the
raw lake inside the football-data.co.uk CSVs. Pinnacle is the sharpest book, and
the CLOSING line (just before kickoff) is its most informed price - the hardest
possible benchmark. Seasons where those columns are absent or sparse are skipped
with a warning rather than silently filled.

Removing the overround: raw implied probabilities `1/odds` sum to more than 1 -
the bookmaker's margin (typically ~2-3% for Pinnacle). They are rescaled to sum
to 1 (proportional normalisation). This is the standard first-order correction;
it slightly over-states long-shot probabilities relative to more elaborate
methods (Shin's, or a power adjustment), which remains a possible refinement.

Joining to our dataset: our models are trained on Understat (which carries xG),
while the odds live in football-data.co.uk. A fixture is keyed by
(season, home_team, away_team) - unique, since each ordered pairing occurs once
per season - avoiding any reliance on matching kickoff timestamps across
sources. Only 7 of 35 club names differ between the two sources; they are mapped
explicitly, and an unmapped name raises rather than silently dropping a match.
"""

import logging

import numpy as np
import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version
from research.data.csv_utils import read_csv_resilient

logger = logging.getLogger(__name__)

SOURCE_NAME = "football_data_co_uk"

# Pinnacle CLOSING odds: home / draw / away. The sharpest public forecast.
ODDS_COLUMNS = ["PSCH", "PSCD", "PSCA"]
# Pinnacle OPENING odds - the price before the market has absorbed team news and
# sharp money. The open->close move is the market learning; anticipating that
# move is the professional definition of having an edge.
OPENING_COLUMNS = ["PSH", "PSD", "PSA"]
MIN_COVERAGE = 0.95  # a season must have odds on ~all matches to be usable

# football-data.co.uk name -> Understat canonical name (only the 7 that differ).
TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "QPR": "Queens Park Rangers",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
}


def _season_label(season_code: str) -> str:
    """'1819' -> '2018-19', matching the Understat season label."""
    return f"20{season_code[:2]}-{season_code[2:]}"


def _canonical(name: str) -> str:
    return TEAM_NAME_MAP.get(name.strip(), name.strip())


def implied_probabilities(odds: np.ndarray) -> np.ndarray:
    """(n, 3) decimal odds -> (n, 3) probabilities summing to 1.

    Divides out the bookmaker's overround proportionally.
    """
    inverse = 1.0 / odds
    return inverse / inverse.sum(axis=1, keepdims=True)


def overround(odds: np.ndarray) -> np.ndarray:
    """The bookmaker's margin per match: sum(1/odds) - 1."""
    return (1.0 / odds).sum(axis=1) - 1.0


def _season_csv(raw_data_dir, league_code: str, season_code: str):
    dataset_dir = raw_data_dir / SOURCE_NAME / league_code / season_code
    version = read_latest_version(dataset_dir)
    if version is None:
        return None
    return dataset_dir / version / f"{season_code}.csv"


def load_closing_odds(league_code: str = "E0", map_names: bool = True) -> pd.DataFrame:
    """Return one row per match that has usable closing odds:
    season, home_team, away_team, p_home, p_draw, p_away, overround.

    `map_names=True` (default) converts club names to the Understat canonical
    form, so the odds join onto the xG-based research dataset. Set it False when
    the matches themselves come from football-data.co.uk (e.g. leagues Understat
    does not cover) - then both sides already share the same names and mapping
    would be wrong.
    """
    config = load_config()
    raw = config.raw_data_dir
    name_fn = _canonical if map_names else (lambda n: n.strip())

    frames = []
    for season_code in config.football_data_co_uk.seasons:
        csv_path = _season_csv(raw, league_code, season_code)
        if csv_path is None:
            continue
        raw_df = read_csv_resilient(csv_path).dropna(subset=["HomeTeam"])

        missing = [c for c in ODDS_COLUMNS if c not in raw_df.columns]
        if missing:
            logger.warning(
                "Season %s: closing-odds columns %s absent - skipping season",
                season_code, missing,
            )
            continue
        coverage = raw_df[ODDS_COLUMNS].notna().all(axis=1).mean()
        if coverage < MIN_COVERAGE:
            logger.warning(
                "Season %s: closing odds present on only %.0f%% of matches - skipping",
                season_code, 100 * coverage,
            )
            continue

        df = raw_df.dropna(subset=ODDS_COLUMNS).copy()
        df = df[(df[ODDS_COLUMNS] > 1.0).all(axis=1)]  # guard against 0/1.0 sentinels

        odds = df[ODDS_COLUMNS].to_numpy(dtype=float)
        probs = implied_probabilities(odds)

        frames.append(
            pd.DataFrame({
                "season": _season_label(season_code),
                # The date is needed as part of the join key: in some leagues
                # (e.g. the Scottish Premiership, which splits mid-season) a team
                # hosts the same opponent TWICE, so (season, home, away) is not
                # unique. Parsed dayfirst, as football-data.co.uk publishes.
                "date": pd.to_datetime(df["Date"], dayfirst=True).to_numpy(),
                "home_team": df["HomeTeam"].map(name_fn).to_numpy(),
                "away_team": df["AwayTeam"].map(name_fn).to_numpy(),
                "p_home": probs[:, 0],
                "p_draw": probs[:, 1],
                "p_away": probs[:, 2],
                "overround": overround(odds),
            })
        )

    if not frames:
        raise ValueError(
            f"No usable closing odds found for league '{league_code}' - has the "
            f"football_data_co_uk source been downloaded?"
        )

    odds_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded closing odds for %d matches across %d seasons (mean overround %.2f%%)",
        len(odds_df), odds_df["season"].nunique(), 100 * odds_df["overround"].mean(),
    )
    return odds_df


def load_pinnacle_odds(league_code: str = "E0") -> pd.DataFrame:
    """Both the OPENING and CLOSING Pinnacle prices for every match that has both.

    Columns: season, home_team, away_team,
             open_p_home/open_p_draw/open_p_away, open_overround,
             close_p_home/close_p_draw/close_p_away, close_overround.

    The open->close movement is the market absorbing information (team news,
    sharp money) between the price going up and kickoff.
    """
    config = load_config()
    raw = config.raw_data_dir
    needed = OPENING_COLUMNS + ODDS_COLUMNS

    frames = []
    for season_code in config.football_data_co_uk.seasons:
        csv_path = _season_csv(raw, league_code, season_code)
        if csv_path is None:
            continue
        raw_df = read_csv_resilient(csv_path).dropna(subset=["HomeTeam"])

        if any(c not in raw_df.columns for c in needed):
            logger.warning("Season %s: opening and/or closing odds absent - skipping", season_code)
            continue
        if raw_df[needed].notna().all(axis=1).mean() < MIN_COVERAGE:
            logger.warning("Season %s: opening/closing odds too sparse - skipping", season_code)
            continue

        df = raw_df.dropna(subset=needed).copy()
        df = df[(df[needed] > 1.0).all(axis=1)]

        open_odds = df[OPENING_COLUMNS].to_numpy(dtype=float)
        close_odds = df[ODDS_COLUMNS].to_numpy(dtype=float)
        open_probs = implied_probabilities(open_odds)
        close_probs = implied_probabilities(close_odds)

        frames.append(
            pd.DataFrame({
                "season": _season_label(season_code),
                "home_team": df["HomeTeam"].map(_canonical).to_numpy(),
                "away_team": df["AwayTeam"].map(_canonical).to_numpy(),
                "open_p_home": open_probs[:, 0],
                "open_p_draw": open_probs[:, 1],
                "open_p_away": open_probs[:, 2],
                "close_p_home": close_probs[:, 0],
                "close_p_draw": close_probs[:, 1],
                "close_p_away": close_probs[:, 2],
                "open_overround": overround(open_odds),
                "close_overround": overround(close_odds),
            })
        )

    if not frames:
        raise ValueError(f"No usable opening+closing odds for league '{league_code}'")

    odds_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded Pinnacle open+close for %d matches, %d seasons "
        "(overround %.2f%% -> %.2f%% as the market sharpens)",
        len(odds_df), odds_df["season"].nunique(),
        100 * odds_df["open_overround"].mean(), 100 * odds_df["close_overround"].mean(),
    )
    return odds_df
