"""Join FPL players to their Understat identities, one season at a time.

The FPL archive gives per-gameweek points and minutes; Understat gives per-match
xG. To use Understat's xG in the FPL backtest, each FPL player must be tied to the
Understat player who is the same human. There is no shared id, so the join is by
name - the exact hazard that has bitten this project before (a wrong match silently
feeds one player's xG into another's projection).

Three defences:

  1. Normalise hard: strip accents, lowercase, drop punctuation. "Bruno Fernandes"
     and "Bruno Miguel Borges Fernandes" collapse toward a common core.
  2. Disambiguate by TEAM. Two "James" at different clubs never collide, because a
     candidate must share the FPL player's club that season.
  3. A hand-checked ALIAS table for the residue - players whose two sources simply
     spell them differently (Sokratis, Emerson, Rui Patricio).

The join is graded, not trusted: `coverage(season)` reports the share of the
*pickable* pool (players with real minutes) that matched, and the caller gates on
it. A season that cannot clear the bar is a season we do not use.
"""

import glob
import json
import logging
import os
import re
import unicodedata
from typing import Dict, Optional

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.metadata_store import read_latest_version
from research.data.fpl_archive import load_gameweeks, load_player_meta
from research.data.fpl_loader import canonical_team

logger = logging.getLogger(__name__)

UNDERSTAT_SOURCE = "understat"
LEAGUE = "EPL"

# FPL full name -> Understat player_name, for players the normaliser cannot reconcile.
# Every entry is a human decision, verified by (season, club) agreement. Keyed on the
# normalised FPL name so accents/case never matter.
ALIASES: Dict[str, str] = {
    "sokratis papastathopoulos": "sokratis",
    "emerson palmieri dos santos": "emerson",
    "rui pedro dos santos patricio": "rui patricio",
    "bernardo mota veiga de carvalho e silva": "bernardo silva",
    "joao pedro cavaco cancelo": "joao cancelo",
    "ricardo domingos barbosa pereira": "ricardo pereira",
    "hee chan hwang": "hwang hee chan",
    "son heung min": "heung-min son",
    "gabriel fernando de jesus": "gabriel jesus",
    "gabriel teodoro martinelli silva": "gabriel martinelli",
    "gabriel dos santos magalhaes": "gabriel",
    "bruno miguel borges fernandes": "bruno fernandes",
    "diogo jose teixeira da silva": "diogo jota",
    "trincao goncalo manuel ganchinho": "francisco trincao",
    "cristian romero": "cristian gabriel romero",
    # Brazilian/Portuguese nicknames that share no token with the FPL legal name.
    "fabio henrique tavares": "fabinho",
    "fernando luiz rosa": "fernandinho",
    "frederico rodrigues de paula santos": "fred",
    "carlos henrique casimiro": "casemiro",
    "bruno andre cavaco jordao": "bruno jordao",
    "jose diogo dalot teixeira": "diogo dalot",
    "joao pedro junqueira de jesus": "joao pedro",
    "estupinan hidalgo pervis josue": "pervis estupinan",
    "vinicius souza": "matheus vinicius",
    "murillo santiago costa dos santos": "murillo",
    "igor julio dos santos de paulo": "igor",
    "douglas luiz soares de paulo": "douglas luiz",
    "carlos vinicius alves morais": "carlos vinicius",
    "matheus santos carneiro da cunha": "matheus cunha",
}


# Letters NFKD does not decompose to ASCII, so they would be DROPPED (turning
# "Ødegaard" into "degaard") and never match. Map them explicitly first.
_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "o", "å": "a", "Å": "a", "æ": "ae", "Æ": "ae",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "ß": "ss",
})


def _norm(name: str) -> str:
    text = str(name).translate(_TRANSLIT)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Separators (hyphens, apostrophes) become spaces, NOT nothing: "Ward-Prowse"
    # must tokenise to ["ward", "prowse"] so its surname can match, not "wardprowse".
    text = re.sub(r"[^a-z]+", " ", text.lower())
    return text.strip()


def _league_dir(start_year: int):
    return load_config().raw_data_dir / UNDERSTAT_SOURCE / LEAGUE / str(start_year)


def _start_year(season: str) -> int:
    return int(season.split("-")[0])


def understat_players(season: str) -> Dict[str, dict]:
    """Understat id -> {name, team} for players who appeared in the EPL that season."""
    directory = _league_dir(_start_year(season))
    version = read_latest_version(directory)
    if version is None:
        raise ValueError(f"No ingested Understat league feed for {season}")
    path = [p for p in glob.glob(os.path.join(str(directory), version, "*.json"))
            if not p.endswith(".meta.json")][0]
    payload = json.loads(open(path, encoding="utf-8").read())
    return {p["id"]: {"name": p["player_name"], "team": canonical_team(p["team_title"])}
            for p in payload["players"]}


def _tokens(name: str) -> list:
    return [t for t in _norm(name).split() if t]


def _match_score(fpl_tokens: list, us_tokens: list) -> int:
    """How strongly two names refer to the same person. Higher is better; 0 = no.

    The two sources spell the same player very differently - full legal name vs
    common name ("Alisson Ramses Becker" vs "Alisson", "Benjamin Chilwell" vs "Ben
    Chilwell", "Bamidele Alli" vs "Dele Alli"). Surnames are the reliable anchor, so
    a shared last token is weighted heavily; shared or abbreviated forenames add a
    little. Scoring is done only among players ALREADY on the same club that season,
    which is what makes a loose token match safe.
    """
    fpl_set, us_set = set(fpl_tokens), set(us_tokens)
    shared = fpl_set & us_set
    score = 3 * len(shared)
    if fpl_tokens and us_tokens and fpl_tokens[-1] == us_tokens[-1]:
        score += 5                                  # same surname: the strong signal
    # A single-name player (Understat often lists Brazilians/Portuguese by one common
    # name) whose sole token appears in the other's full name, same club: near-certain.
    if len(us_tokens) == 1 and us_tokens[0] in fpl_set:
        score += 5
    if len(fpl_tokens) == 1 and fpl_tokens[0] in us_set:
        score += 5
    # Abbreviated forename: "ben" for "benjamin", "alex" for "alexandre".
    for a in fpl_set:
        for b in us_set:
            if a != b and (a.startswith(b) or b.startswith(a)) and min(len(a), len(b)) >= 3:
                score += 1
    return score


def build_map(season: str) -> Dict[int, str]:
    """FPL element id -> Understat id, for one season. Unmatched players are absent.

    Exact (name, team) first; then, within the same club, the best token match by a
    strict margin. Confining the fuzzy match to one club is the safeguard - two
    players who share a name almost never share a team.
    """
    us = understat_players(season)

    by_name_team = {}
    us_by_team = {}
    for uid, info in us.items():
        by_name_team[(_norm(info["name"]), info["team"])] = uid
        us_by_team.setdefault(info["team"], []).append((uid, _tokens(info["name"])))

    fpl_meta = load_player_meta(season)
    mapping, used = {}, set()
    # Deterministic order so the greedy claim of an Understat id is reproducible.
    for element in sorted(fpl_meta):
        meta = fpl_meta[element]
        team = meta["team"]
        name = _norm(meta["name"])
        alias = ALIASES.get(name)

        # 1. Exact name+team (also via alias).
        matched = by_name_team.get((name, team)) or (
            by_name_team.get((alias, team)) if alias else None)

        # 2. Best in-club token match, if unambiguous.
        if matched is None and team in us_by_team:
            fpl_tokens = _tokens(alias) if alias else _tokens(meta["name"])
            scored = sorted(
                ((_match_score(fpl_tokens, us_tokens), uid)
                 for uid, us_tokens in us_by_team[team] if uid not in used),
                reverse=True,
            )
            if scored and scored[0][0] >= 5 and (
                    len(scored) == 1 or scored[0][0] > scored[1][0]):
                matched = scored[0][1]

        if matched is not None:
            mapping[element] = matched
            used.add(matched)

    # Second pass, team-agnostic, for mid-season transfers: FPL records a player at
    # his end-of-season club, Understat aggregates him under one club, and the two
    # disagree - so the in-club match above cannot see him. Only an EXACT normalised
    # full-name match, unique among still-unused Understat ids, is accepted here; no
    # team safeguard means the bar has to be identity, not resemblance.
    exact_us = {}
    for uid, info in us.items():
        exact_us.setdefault(_norm(info["name"]), []).append(uid)
    for element in sorted(fpl_meta):
        if element in mapping:
            continue
        name = _norm(fpl_meta[element]["name"])
        candidates = [u for u in exact_us.get(name, []) if u not in used]
        if len(candidates) == 1:
            mapping[element] = candidates[0]
            used.add(candidates[0])
    return mapping


def coverage(season: str) -> dict:
    """How much of the season matched, overall and on the pool that matters.

    The pickable pool is players who accumulated real minutes; a projection is only
    ever built from those, so their coverage is the number that gates the season.
    """
    mapping = build_map(season)
    gw = load_gameweeks(season)
    minutes = gw.groupby("player_id")["minutes"].sum()

    pickable = minutes[minutes >= 900].index          # ~10 full matches
    everyone = minutes[minutes > 0].index

    def share(ids):
        ids = list(ids)
        if not ids:
            return float("nan")
        return sum(1 for e in ids if e in mapping) / len(ids)

    return {
        "season": season,
        "fpl_players_with_minutes": int((minutes > 0).sum()),
        "matched_overall": share(everyone),
        "pickable_players": int(len(pickable)),
        "matched_pickable": share(pickable),
        "unmatched_pickable": sorted(
            gw[gw["player_id"] == e]["player"].iloc[0] for e in pickable if e not in mapping
        ),
    }
