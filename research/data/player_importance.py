"""Player importance, derived from the PREVIOUS season's minutes played.

Phase 3b showed a raw count of absences is too blunt: it weights a missing star
striker the same as a missing third-choice full-back. This module supplies the
missing ingredient - how important each player is - so an absence can be
weighted rather than merely counted.

Importance = minutes played in the PREVIOUS season / a full season of minutes
(38 x 90 = 3420), clipped to [0, 1]. So a 1.0 means "an ever-present regular",
0.1 means "a fringe player", and 0.0 means "did not play in this league last
season" (a youth player or a brand-new signing).

Why the *previous* season and not the current one: a player's current-season
minutes include matches played AFTER the fixture being predicted. Using them
would leak the future into the feature. Previous-season minutes are entirely in
the past at kickoff, so they are leakage-safe by construction.

Source: the Understat league payload already in the raw lake carries per-player
season stats (`players` -> player_name, time, team_title). No extra API calls
and no paid data are needed.

Name matching: injury records (API-Football) abbreviate the first name
("D. de Gea") while Understat spells it out ("David de Gea"). Both are reduced
to a (first-initial, surname) key after stripping accents and case. If that
misses, a surname-only fallback is used when the surname is unambiguous in the
league. Anything still unmatched scores 0.0 - which is the correct default,
since a player absent from last season's minutes table genuinely had no minutes.
"""

import json
import logging
import re
import unicodedata
from typing import Dict, Optional, Tuple

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version

logger = logging.getLogger(__name__)

SOURCE_NAME = "understat"
MAX_SEASON_MINUTES = 38 * 90  # 3420 - an ever-present outfield/keeper season


def _strip_accents(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def name_key(name: str) -> Optional[Tuple[str, str]]:
    """'David de Gea' and 'D. de Gea' both -> ('d', 'gea').

    Single-token names ('Fabinho') -> ('', 'fabinho'). Returns None for names
    that reduce to nothing.
    """
    cleaned = _strip_accents(name).lower().strip()
    cleaned = re.sub(r"[^a-z\s.'-]", "", cleaned)
    # Hyphens/apostrophes are kept (Rak-Sakyi, O'Brien) but a token must contain
    # at least one letter, so punctuation-only input yields no key at all.
    tokens = [t for t in re.split(r"\s+", cleaned) if re.search(r"[a-z]", t)]
    if not tokens:
        return None
    if len(tokens) == 1:
        return ("", tokens[0])
    return (tokens[0][0], tokens[-1])


class ImportanceLookup:
    """Maps a player name to their previous-season importance in [0, 1]."""

    def __init__(self, minutes_by_key: Dict[Tuple[str, str], int], minutes_by_surname: Dict[str, list]):
        self._by_key = minutes_by_key
        self._by_surname = minutes_by_surname

    def minutes(self, player_name: str) -> int:
        key = name_key(player_name)
        if key is None:
            return 0
        if key in self._by_key:
            return self._by_key[key]
        # Fallback: unambiguous surname (handles e.g. a differently-rendered
        # first name). If several players share the surname, refuse to guess.
        candidates = self._by_surname.get(key[1], [])
        if len(candidates) == 1:
            return candidates[0]
        return 0

    def importance(self, player_name: str) -> float:
        return min(self.minutes(player_name) / MAX_SEASON_MINUTES, 1.0)


def _season_players(league: str, af_season: str) -> list:
    config = load_config()
    dataset_dir = config.raw_data_dir / SOURCE_NAME / league / af_season
    version = read_latest_version(dataset_dir)
    if version is None:
        raise ValueError(
            f"No ingested Understat data for {league}/{af_season} - needed for "
            f"player importance (previous-season minutes)."
        )
    path = dataset_dir / version / f"{league}_{af_season}.json"
    return json.loads(path.read_text(encoding="utf-8"))["players"]


def build_importance_lookup(af_season: str, league: str = "EPL") -> ImportanceLookup:
    """Build the importance lookup from the minutes played in `af_season`
    (which the caller supplies as the season BEFORE the one being predicted)."""
    by_key: Dict[Tuple[str, str], int] = {}
    by_surname: Dict[str, list] = {}

    for player in _season_players(league, af_season):
        minutes = int(player["time"])
        key = name_key(player["player_name"])
        if key is None:
            continue
        # Two players can reduce to the same key (rare); keep the more
        # prominent one, since that is the absence that would matter.
        by_key[key] = max(by_key.get(key, 0), minutes)
        # One entry per player, so the surname fallback can tell "exactly one
        # player has this surname" (safe to use) from "several do" (refuse).
        by_surname.setdefault(key[1], []).append(minutes)

    logger.info(
        "Built importance lookup from %s: %d players", af_season, len(by_key)
    )
    return ImportanceLookup(by_key, by_surname)
