"""Resolve archived predicted-lineup names to FPL player ids.

The archiver (`predicted_lineups.py`) stores names verbatim, on purpose: the archive
must stay faithful, so that a better matcher later can re-resolve every historical
snapshot. This is that matcher — the bridge from a source's free-text names to the
`Start %` per FPL player id that the minutes model can actually consume.

The join is genuinely awkward, and each hazard below is real, taken from the live feed:

  concatenation   "KepaArrizabalaga" — the source drops the space. Splitting on camel
                  case would break "McTominay" instead, so nothing is split: names are
                  compared *squashed* (spaces removed) as well as normally, which
                  handles both without guessing.
  accents         the source strips them ("Gyokeres", "Odegaard", "Magalhaes"); FPL
                  keeps them ("Gyökeres", "Ødegaard", "Magalhães"). NFKD fixes most,
                  but Ø/æ/ß/ł are distinct letters that do NOT decompose, so they are
                  mapped explicitly.
  short names     "David Raya" vs FPL's "David Raya Martin"; "Gabriel Magalhaes" vs
                  "Gabriel dos Santos Magalhães". Token-subset matching handles these,
                  where an exact compare cannot.
  ambiguity       Arsenal alone fields Gabriel (Magalhães), Gabriel Jesus and Gabriel
                  Martinelli. Matching is therefore scoped to one club, which cuts the
                  candidate pool to ~25 and makes a first name insufficient on its own.

Every unmatched player is reported, never silently dropped: a player we fail to
resolve is a player whose rotation risk we simply do not see, and that must be visible
rather than quietly absorbed.
"""

import difflib
import logging
import re
import unicodedata
from typing import Dict, List, Optional

import pandas as pd

from research.data.fpl_loader import canonical_team

logger = logging.getLogger(__name__)

# Letters that are NOT accents and so survive NFKD untouched, but which the source
# writes as their plain ASCII cousin.
_SPECIAL_LETTERS = {
    "ø": "o", "æ": "ae", "œ": "oe", "ß": "ss", "đ": "d", "ð": "d",
    "ł": "l", "þ": "th", "ı": "i", "ħ": "h", "ŋ": "n",
}

# Below this, a "best match" is a guess, and a wrong id is worse than a known gap:
# it would silently attribute one player's rotation risk to another.
MATCH_THRESHOLD = 0.80


def normalize(name: str) -> str:
    """Lowercase, accent-stripped, punctuation-free, whitespace-collapsed."""
    text = unicodedata.normalize("NFKD", (name or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(_SPECIAL_LETTERS.get(ch, ch) for ch in text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def squash(name: str) -> str:
    """Normalized with spaces removed — collapses "Kepa Arrizabalaga" and
    "KepaArrizabalaga" onto the same key without ever splitting a name."""
    return normalize(name).replace(" ", "")


def _candidate(player: pd.Series) -> dict:
    full = normalize(player.get("full_name", ""))
    web = normalize(player.get("web_name", ""))
    return {
        "id": int(player["id"]),
        "full_norm": full,
        "full_sq": full.replace(" ", ""),
        "full_tokens": set(full.split()),
        "web_norm": web,
        "web_sq": web.replace(" ", ""),
    }


def _score(name: str, candidate: dict):
    """(score, method) for one candidate. Ordered most- to least-certain."""
    norm = normalize(name)
    sq = norm.replace(" ", "")
    tokens = set(norm.split())

    if sq and sq == candidate["full_sq"]:
        return 1.0, "exact_full"
    if sq and sq == candidate["web_sq"]:
        return 0.97, "exact_web"
    # "David Raya" ⊂ "David Raya Martin". Requires >1 token so a bare first name
    # ("Gabriel") cannot claim a match on its own.
    if len(tokens) > 1 and tokens <= candidate["full_tokens"]:
        return 0.93, "token_subset"
    # FPL's web_name is very often just the surname.
    if tokens and norm.split()[-1] == candidate["web_norm"]:
        return 0.90, "surname_web"
    return difflib.SequenceMatcher(None, norm, candidate["full_norm"]).ratio(), "fuzzy"


def resolve_name(name: str, candidates: List[dict],
                 threshold: float = MATCH_THRESHOLD):
    """Best (player_id, score, method) for `name`, or (None, score, method) if the
    best candidate is too weak to trust."""
    best_id, best_score, best_method = None, 0.0, "none"
    for candidate in candidates:
        score, method = _score(name, candidate)
        if score > best_score:
            best_id, best_score, best_method = candidate["id"], score, method
    if best_score >= threshold:
        return best_id, best_score, best_method

    # Last resort for the source's concatenation glitch, where every rule above is
    # structurally unable to fire: "KepaArrizabalaga" is ONE token (so token_subset
    # cannot apply) and FPL carries an extra surname ("Kepa Arrizabalaga Revuelta"),
    # so the squashed forms are not equal either — only a prefix.
    #
    # Gated on UNIQUENESS rather than on a length guess: a prefix is accepted only if
    # exactly one player in the club can claim it. That is what stops a bare "Gabriel"
    # from silently becoming one of Arsenal's three.
    squashed = squash(name)
    if squashed:
        prefixed = [c for c in candidates
                    if c["full_sq"].startswith(squashed) or squashed.startswith(c["full_sq"])]
        if len(prefixed) == 1:
            return prefixed[0]["id"], 0.92, "squash_prefix"

    return None, best_score, best_method


def resolve_snapshot(snapshot: dict, players: pd.DataFrame,
                     threshold: float = MATCH_THRESHOLD) -> dict:
    """Map an archived snapshot onto FPL player ids.

    Returns `start_pct` ({player_id: 0-100}) for the minutes model, plus the matched
    and — importantly — the **unmatched** rows, so a silent gap is impossible.
    Matching is scoped per club, which is what makes ambiguous first names tractable.
    """
    by_team = {}
    for team, group in players.groupby("team"):
        by_team[team] = [_candidate(row) for _, row in group.iterrows()]

    start_pct: Dict[int, int] = {}
    matched, unmatched, unknown_teams = [], [], []

    for team_block in snapshot.get("teams", []):
        engine_team = canonical_team(team_block.get("canonical_team", team_block["team"]))
        candidates = by_team.get(engine_team)
        if not candidates:
            unknown_teams.append(engine_team)
            logger.warning("no FPL squad for team %r — skipping %d players",
                           engine_team, len(team_block.get("players", [])))
            continue

        for player in team_block.get("players", []):
            player_id, score, method = resolve_name(player["name"], candidates, threshold)
            record = {
                "team": engine_team,
                "name": player["name"],
                "position": player.get("position"),
                "start_pct": player["start_pct"],
                "score": round(float(score), 3),
                "method": method,
            }
            if player_id is None:
                unmatched.append(record)
                continue
            record["player_id"] = player_id
            matched.append(record)
            # A player can appear in both the XI and the alternatives table; keep the
            # higher probability rather than whichever happened to parse last.
            start_pct[player_id] = max(start_pct.get(player_id, 0), int(player["start_pct"]))

    total = len(matched) + len(unmatched)
    stats = {
        "teams": len(snapshot.get("teams", [])),
        "players": total,
        "matched": len(matched),
        "unmatched": len(unmatched),
        "match_rate": round(len(matched) / total, 4) if total else 0.0,
        "unknown_teams": sorted(set(unknown_teams)),
    }
    if unmatched:
        logger.warning("%d/%d names unresolved (%.1f%% matched)",
                       len(unmatched), total, 100 * stats["match_rate"])
    return {"start_pct": start_pct, "matched": matched,
            "unmatched": unmatched, "stats": stats}
