"""Loads the FPL bootstrap payload from the raw lake into clean DataFrames.

FPL's `bootstrap-static` is one large JSON blob. This turns it into two tables -
players and teams - with only the fields a projection needs, typed properly, and
with per-90 rates precomputed (the form in which every projection consumes them).

FPL names its own teams ("Man City", "Spurs"); the prediction engine is trained
on Understat's names ("Manchester City", "Tottenham"). The six that differ are
mapped explicitly, and `unmapped_teams()` fails loudly rather than silently
dropping a club.
"""

import json
import logging
from typing import List, Optional

import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version

logger = logging.getLogger(__name__)

SOURCE_NAME = "fpl"
MINUTES_PER_MATCH = 90

# FPL team name -> Understat canonical name (only the ones that differ).
TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",
    "Spurs": "Tottenham",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
}

# Availability: FPL marks 'a' available; anything else is doubt/injury/suspension.
AVAILABLE_STATUS = "a"


def _load_dataset(dataset: str) -> dict:
    config = load_config()
    dataset_dir = config.raw_data_dir / SOURCE_NAME / dataset
    version = read_latest_version(dataset_dir)
    if version is None:
        raise ValueError(
            f"No ingested FPL '{dataset}' - run: python -m data_warehouse.cli "
            f"download --source fpl"
        )
    path = dataset_dir / version / f"{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_team(fpl_name: str) -> str:
    return TEAM_NAME_MAP.get(fpl_name, fpl_name)


def load_teams() -> pd.DataFrame:
    payload = _load_dataset("bootstrap-static")
    return pd.DataFrame([
        {
            "fpl_id": t["id"],
            "fpl_name": t["name"],
            "team": canonical_team(t["name"]),
            "short_name": t["short_name"],
        }
        for t in payload["teams"]
    ])


def load_players() -> pd.DataFrame:
    """One row per player, with per-90 rates ready for projection."""
    payload = _load_dataset("bootstrap-static")
    teams = {t["id"]: canonical_team(t["name"]) for t in payload["teams"]}

    rows = []
    for element in payload["elements"]:
        minutes = int(element["minutes"])
        per_90 = minutes / MINUTES_PER_MATCH if minutes > 0 else 0.0
        rows.append({
            "id": element["id"],
            # FPL's permanent player code. `id` is reassigned every season, so `code`
            # is the only safe key for joining a previous season's history onto this
            # season's squad.
            "code": int(element["code"]),
            "web_name": element["web_name"],
            "full_name": f"{element['first_name']} {element['second_name']}".strip(),
            "team": teams[element["team"]],
            "position": int(element["element_type"]),
            "price": int(element["now_cost"]) / 10.0,
            "minutes": minutes,
            "starts": int(element["starts"]),
            "total_points": int(element["total_points"]),
            "goals_scored": int(element["goals_scored"]),
            "assists": int(element["assists"]),
            "saves": int(element["saves"]),
            "bonus": int(element["bonus"]),
            "yellow_cards": int(element["yellow_cards"]),
            "red_cards": int(element["red_cards"]),
            "defensive_contribution": int(element.get("defensive_contribution", 0) or 0),
            "expected_goals": float(element["expected_goals"]),
            "expected_assists": float(element["expected_assists"]),
            "available": element["status"] == AVAILABLE_STATUS,
            "chance_of_playing": (
                100.0 if element["chance_of_playing_next_round"] is None
                else float(element["chance_of_playing_next_round"])
            ),
            # Per-90 rates: the natural unit for projecting a single fixture.
            "xg_per_90": float(element["expected_goals"]) / per_90 if per_90 else 0.0,
            "xa_per_90": float(element["expected_assists"]) / per_90 if per_90 else 0.0,
            "saves_per_90": int(element["saves"]) / per_90 if per_90 else 0.0,
            "bonus_per_90": int(element["bonus"]) / per_90 if per_90 else 0.0,
            "dc_per_90": (int(element.get("defensive_contribution", 0) or 0) / per_90) if per_90 else 0.0,
            "cards_per_90": (
                (int(element["yellow_cards"]) + 3 * int(element["red_cards"])) / per_90
                if per_90 else 0.0
            ),
        })

    players = pd.DataFrame(rows)
    logger.info(
        "Loaded %d FPL players across %d teams", len(players), players["team"].nunique()
    )
    return players


def unmapped_teams(engine_teams: List[str]) -> List[str]:
    """FPL teams whose canonical name is not known to the prediction engine.
    Empty list means the join is complete."""
    known = set(engine_teams)
    return sorted(t for t in load_teams()["team"] if t not in known)


def load_fixtures() -> pd.DataFrame:
    """Every fixture with its gameweek, teams (canonical names) and status.

    From FPL's `fixtures` endpoint. `gameweek` is None for a fixture not yet
    assigned to a gameweek. Team ids are resolved through the same canonical
    mapping the projection engine is trained on, so `home_team`/`away_team` join
    straight onto it.
    """
    payload = _load_dataset("fixtures")
    teams = {t["id"]: canonical_team(t["name"]) for t in _load_dataset("bootstrap-static")["teams"]}
    rows = []
    for fixture in payload:
        rows.append({
            "fixture_id": fixture["id"],
            "gameweek": fixture["event"],          # None until scheduled
            "home_team": teams[fixture["team_h"]],
            "away_team": teams[fixture["team_a"]],
            "kickoff_time": fixture.get("kickoff_time"),
            "finished": bool(fixture.get("finished")),
        })
    return pd.DataFrame(rows)


def next_gameweek() -> Optional[dict]:
    """The upcoming gameweek as {gameweek, deadline_time}, or None off-season.

    Prefers FPL's own `is_next` flag; falls back to the earliest gameweek not yet
    finished. Returns None when every gameweek is finished (the game is between
    seasons and the next one is not yet published).
    """
    events = _load_dataset("bootstrap-static")["events"]
    for event in events:
        if event.get("is_next"):
            return {"gameweek": int(event["id"]), "deadline_time": event["deadline_time"]}
    upcoming = [e for e in events if not e.get("finished")]
    if not upcoming:
        return None
    event = min(upcoming, key=lambda e: e["deadline_time"])
    return {"gameweek": int(event["id"]), "deadline_time": event["deadline_time"]}


def fixtures_for_gameweek(gameweek: int) -> List[tuple]:
    """The (home_team, away_team) pairs scheduled for `gameweek`, canonical names."""
    fixtures = load_fixtures()
    week = fixtures[fixtures["gameweek"] == gameweek]
    return [(row["home_team"], row["away_team"]) for _, row in week.iterrows()]
