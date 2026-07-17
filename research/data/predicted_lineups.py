"""Archive pre-deadline PREDICTED lineups — the dataset nobody kept.

Phase 6f measured the prize: perfect knowledge of who actually plays is worth
~+4.6 pts/GW on top of the recent-form minutes model (`docs/02_FEATURE_ROADMAP.md`).
Predicted lineups are the only pre-deadline route to a slice of that. But they
**cannot be backtested**, for one banal reason: nobody archived them. You cannot go
back and ask what a predicted XI *was* last Saturday.

So this module archives them, weekly, from now on. Two payoffs:
  - Bank-It can consume them live from GW1;
  - after a season we OWN the historical series that does not exist today, and the
    signal becomes properly backtestable instead of forever taken on faith.

Why this source: FPL's deadline is 90 minutes before the gameweek's FIRST kickoff,
while official lineups appear ~1 hour before EACH kickoff — so confirmed lineups are
never available in time (a Sunday XI is confirmed a day *after* the deadline). Only
*fantasy*-oriented predictions, published days ahead for the whole gameweek, are
usable. Fantasy Football Pundit publishes exactly that, free, for all 20 clubs, and —
critically — as a **per-player Start %** rather than a binary XI, which drops straight
into the minutes model's p_play/p_60 instead of needing a lossy conversion.

Politeness: their robots.txt allows all paths with `Crawl-delay: 10`. This fetches a
single page per gameweek, so the delay is honoured by construction; `MIN_FETCH_GAP`
enforces it anyway if ever called in a loop.

Deliberately does NOT resolve names to FPL player ids. Archive faithfully, resolve at
consumption time — so a better matcher later can re-resolve every historical snapshot.
"""

import html as html_module
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

SOURCE = "fantasyfootballpundit"
SOURCE_URL = "https://www.fantasyfootballpundit.com/fantasy-premier-league-team-news/"

# Bumped whenever the parser's output shape or semantics change, so a snapshot is
# always interpretable years later.
PARSER_VERSION = 1

# robots.txt: `Crawl-delay: 10`.
MIN_FETCH_GAP_SECONDS = 10
_LAST_FETCH = [0.0]

# The site 403s an obvious bot agent, as FPL's own endpoint does.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "lineups"

_HEADING_RE = re.compile(r"(?is)<h[1-4][^>]*>.*?</h[1-4]>")
_TABLE_RE = re.compile(r"(?is)<table[^>]*>.*?</table>")
_ROW_RE = re.compile(r"(?is)<tr>(.*?)</tr>")
_CELL_RE = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")
_TEAM_HEADING_RE = re.compile(r"^(?P<team>.+?)\s+Predicted Lineup\s*$", re.I)
_PCT_RE = re.compile(r"(\d{1,3})\s*%")

# FFP spells a few clubs its own way. Map to the FPL/engine canonical names; anything
# not listed already matches. `research.data.fpl_loader.canonical_team` then carries
# it the rest of the way to the engine's Understat naming.
TEAM_NAME_FIXES = {
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Spurs",
}


def _text(fragment: str) -> str:
    """Tag-stripped, entity-decoded, whitespace-collapsed text."""
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def fetch_html(url: str = SOURCE_URL, timeout: int = 30) -> str:
    """Fetch the team-news page, honouring the published crawl delay."""
    gap = time.time() - _LAST_FETCH[0]
    if gap < MIN_FETCH_GAP_SECONDS:
        time.sleep(MIN_FETCH_GAP_SECONDS - gap)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    _LAST_FETCH[0] = time.time()
    response.raise_for_status()
    return response.text


def parse(page: str) -> List[dict]:
    """Every club's predicted lineup, as [{team, players: [...]}].

    Each club heading (`"Arsenal Predicted Lineup"`) is followed by two tables: the
    predicted XI (header `Player`) and the alternatives (header `Potential Starters`).
    Both carry a per-player `Start %`, so both are kept — a 40%-to-start alternative is
    exactly the rotation signal we are here for. `predicted_xi` records which table a
    player came from.
    """
    clean = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", page)

    headings = []
    for match in _HEADING_RE.finditer(clean):
        team_match = _TEAM_HEADING_RE.match(_text(match.group(0)))
        if team_match:
            headings.append((match.start(), team_match.group("team").strip()))

    teams = {}
    order = []
    for table in _TABLE_RE.finditer(clean):
        body = table.group(0)
        if "Start %" not in body:
            continue
        prior = [name for pos, name in headings if pos < table.start()]
        if not prior:
            continue
        team = prior[-1]

        rows = [[_text(cell) for cell in _CELL_RE.findall(row)]
                for row in _ROW_RE.findall(body)]
        if not rows:
            continue
        header = rows[0]
        is_xi = bool(header) and header[0].strip().lower() == "player"

        players = []
        for cells in rows[1:]:
            if len(cells) < 3:
                continue
            name, position, pct = cells[0], cells[1], _PCT_RE.search(cells[2])
            if not name or pct is None:
                continue
            players.append({
                "name": name,
                "position": position,
                "start_pct": int(pct.group(1)),
                "predicted_xi": is_xi,
            })
        if not players:
            continue
        if team not in teams:
            teams[team] = []
            order.append(team)
        teams[team].extend(players)

    return [{"team": team,
             "canonical_team": TEAM_NAME_FIXES.get(team, team),
             "players": teams[team]} for team in order]


def build_snapshot(page: str, season: str, gameweek: Optional[int] = None,
                   deadline_time: Optional[str] = None) -> dict:
    teams = parse(page)
    return {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "parser_version": PARSER_VERSION,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "gameweek": gameweek,
        "deadline_time": deadline_time,
        "teams": teams,
        "team_count": len(teams),
        "player_count": sum(len(t["players"]) for t in teams),
    }


def snapshot_path(season: str, gameweek: Optional[int], fetched_at: str,
                  archive_dir: Path = ARCHIVE_DIR) -> Path:
    stamp = fetched_at.replace(":", "").replace("-", "")
    label = "GW%02d" % gameweek if gameweek is not None else "preseason"
    return archive_dir / season / ("%s_%s.json" % (label, stamp))


def save_snapshot(snapshot: dict, archive_dir: Path = ARCHIVE_DIR) -> Path:
    path = snapshot_path(snapshot["season"], snapshot["gameweek"],
                         snapshot["fetched_at"], archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("archived %d teams / %d players to %s",
                snapshot["team_count"], snapshot["player_count"], path)
    return path


def current_season(today: datetime = None) -> str:
    """The football season label for a date: August-onwards starts a new one."""
    today = today or datetime.now(timezone.utc)
    start = today.year if today.month >= 7 else today.year - 1
    return "%d-%s" % (start, str(start + 1)[-2:])


EXPECTED_TEAMS = 20


def archive_now(season: str = None, gameweek: int = None,
                archive_dir: Path = ARCHIVE_DIR) -> dict:
    """Fetch, parse and archive one pre-deadline snapshot.

    `gameweek` None auto-detects the upcoming gameweek; between seasons no gameweek is
    published, and the snapshot is still archived (labelled `preseason`) because the
    timestamp is what matters — it can be attributed to a gameweek later.
    """
    season = season or current_season()
    deadline = None
    if gameweek is None:
        try:
            from research.data.fpl_loader import next_gameweek
            upcoming = next_gameweek()
            if upcoming:
                gameweek, deadline = upcoming["gameweek"], upcoming["deadline_time"]
        except Exception as exc:                     # never lose a snapshot over metadata
            logger.warning("could not read the upcoming gameweek (%s); archiving anyway", exc)

    snapshot = build_snapshot(fetch_html(), season, gameweek=gameweek,
                              deadline_time=deadline)
    if snapshot["team_count"] < EXPECTED_TEAMS:
        # Loud, but never fatal: a half-parsed snapshot still beats no snapshot, and
        # the raw counts are recorded so a re-parse can spot it later.
        logger.warning("parsed only %d/%d teams — the page layout may have changed",
                       snapshot["team_count"], EXPECTED_TEAMS)
    save_snapshot(snapshot, archive_dir)
    return snapshot


def main(argv: List[str] = None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="research.data.predicted_lineups",
        description="Archive a pre-deadline predicted-lineup snapshot.")
    parser.add_argument("--season", default=None, help="e.g. 2026-27 (default: current)")
    parser.add_argument("--gameweek", type=int, default=None,
                        help="target gameweek (default: the upcoming one)")
    args = parser.parse_args(argv)

    snapshot = archive_now(season=args.season, gameweek=args.gameweek)
    print("archived %s GW=%s — %d teams, %d players (%s)" % (
        snapshot["season"], snapshot["gameweek"], snapshot["team_count"],
        snapshot["player_count"], snapshot["fetched_at"]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
