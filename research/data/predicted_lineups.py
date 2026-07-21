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

# The gameweek label is read straight from FPL's own public endpoint rather than the
# ingested raw lake, so the archiver is STANDALONE: `requests` and nothing else. That
# is what lets it run in CI (where no data is ingested and pandas is not installed) —
# and CI is the only scheduler that cannot be defeated by a closed laptop.
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# Bumped whenever the parser's output shape or semantics change, so a snapshot is
# always interpretable years later. v2: tolerant in-season headings (opponent/gameweek
# suffix, spelling variants) and a per-team `health` block on every snapshot.
PARSER_VERSION = 2

# robots.txt: `Crawl-delay: 10`.
MIN_FETCH_GAP_SECONDS = 10
_LAST_FETCH = [0.0]

# The site intermittently serves a 200 that is not the real page, so a fetch is only
# believed once this marker appears. Retries stay within the crawl delay.
LINEUP_MARKER = "Start %"
FETCH_ATTEMPTS = 3

# The site 403s an obvious bot agent, as FPL's own endpoint does.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

ARCHIVE_DIR = Path(__file__).resolve().parents[1] / "data" / "lineups"

_HEADING_RE = re.compile(r"(?is)<h[1-4][^>]*>.*?</h[1-4]>")
_TABLE_RE = re.compile(r"(?is)<table[^>]*>.*?</table>")
_ROW_RE = re.compile(r"(?is)<tr>(.*?)</tr>")
_CELL_RE = re.compile(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>")
# Deliberately tolerant of in-season cosmetics. The archiver has only ever seen
# PRESEASON pages, where the heading is exactly "Arsenal Predicted Lineup". Once
# fixtures exist the site is very likely to append the opponent or gameweek
# ("Arsenal Predicted Lineup vs Chelsea", "... (GW20)") or vary the spelling
# ("Line-up", "XI"). The old anchored `...Predicted Lineup$` matched NONE of those and
# would have yielded 0 teams — a silent, unrecoverable hole in the archive on the very
# first real gameweek. This captures the team before the marker and ignores whatever
# follows; it still refuses the section header ("Predicted Starting Lineups"), because
# "Predicted" must be immediately followed by "Line…"/"XI", not "Starting".
_TEAM_HEADING_RE = re.compile(r"^(?P<team>.+?)\s+Predicted\s+(?:Line[\s-]?up|XI)\b", re.I)
_PCT_RE = re.compile(r"(\d{1,3})\s*%")

# A predicted XI is eleven names. A team that parses with fewer lost even its starting
# eleven — a partial parse, not a thin news day. Used to distinguish structural drift
# (which must fail loudly) from a legitimately smaller slate (a blank/partial gameweek,
# where fewer teams play but each still parses a full XI).
MIN_PLAYERS_PER_TEAM = 11

# FFP spells a few clubs its own way. Map to the FPL/engine canonical names; anything
# not listed already matches. `research.data.fpl_loader.canonical_team` then carries
# it the rest of the way to the engine's Understat naming.
TEAM_NAME_FIXES = {
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham": "Spurs",
}


class LineupFetchError(RuntimeError):
    """The page came back, but carried no lineups — a failed fetch in disguise."""


def _text(fragment: str) -> str:
    """Tag-stripped, entity-decoded, whitespace-collapsed text."""
    return re.sub(r"\s+", " ", html_module.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def diagnose(page: str) -> str:
    """Why a page yielded nothing — enough to tell a bot-challenge/consent wall from a
    genuine layout change without needing the payload itself."""
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", page)
    return ("bytes=%d title=%r has_start_pct=%s has_heading=%s has_table=%s"
            % (len(page),
               _text(title.group(1))[:90] if title else None,
               "Start %" in page,
               "Predicted Lineup" in page,
               "<table" in page.lower()))


def fetch_html(url: str = SOURCE_URL, timeout: int = 30,
               attempts: int = FETCH_ATTEMPTS) -> str:
    """Fetch the team-news page, honouring the published crawl delay.

    Retries while the response carries no lineups. The site intermittently answers
    **200 with a challenge/consent page** instead of the real one — observed directly:
    two CI runs three minutes apart returned 0 teams and then 20. A transient like that
    would otherwise punch an unfillable hole in the archive on that day, so it is worth
    a couple of extra polite attempts. Returns the last page regardless; the caller
    raises with diagnostics if it still carries nothing.
    """
    page = None
    for attempt in range(1, attempts + 1):
        gap = time.time() - _LAST_FETCH[0]
        if gap < MIN_FETCH_GAP_SECONDS:
            time.sleep(MIN_FETCH_GAP_SECONDS - gap)
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        _LAST_FETCH[0] = time.time()
        response.raise_for_status()
        page = response.text
        if LINEUP_MARKER in page:
            return page
        logger.warning("attempt %d/%d carried no lineups (%s)",
                       attempt, attempts, diagnose(page))
    return page


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


def parse_health(teams: List[dict]) -> dict:
    """Per-team completeness, so a PARTIAL parse cannot pass as a whole one.

    The all-or-nothing checks (0 teams, <20 teams) catch a page that failed entirely.
    They do not catch the subtler in-season risk: the club headings still parse, but a
    change to the *table* structure halves the players per team, so we keep archiving
    snapshots that look fine and quietly carry half the signal. `degraded` is that
    alarm — it fires when the MEDIAN team has lost its starting eleven, which is
    structural drift, not the ordinary variation of a light news day.
    """
    counts = sorted(len(t["players"]) for t in teams)
    n = len(counts)
    median = counts[n // 2] if n else 0
    return {
        "min_players_per_team": counts[0] if n else 0,
        "median_players_per_team": median,
        "max_players_per_team": counts[-1] if n else 0,
        "thin_teams": [t["team"] for t in teams
                       if len(t["players"]) < MIN_PLAYERS_PER_TEAM],
        "degraded": bool(n and median < MIN_PLAYERS_PER_TEAM),
    }


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
        "health": parse_health(teams),
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


def upcoming_gameweek():
    """(gameweek, deadline_time) for the next gameweek, from FPL's public API.

    Returns (None, None) between seasons, or if the endpoint is unreachable — a
    snapshot is never worth losing over a missing label.
    """
    try:
        response = requests.get(FPL_BOOTSTRAP_URL, headers={"User-Agent": USER_AGENT},
                                timeout=30)
        response.raise_for_status()
        events = response.json().get("events", [])
    except Exception as exc:
        logger.warning("could not read the upcoming gameweek (%s); archiving anyway", exc)
        return None, None

    for event in events:
        if event.get("is_next"):
            return int(event["id"]), event.get("deadline_time")
    pending = [e for e in events if not e.get("finished") and e.get("deadline_time")]
    if pending:
        event = min(pending, key=lambda e: e["deadline_time"])
        return int(event["id"]), event.get("deadline_time")
    return None, None


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
        gameweek, deadline = upcoming_gameweek()

    page = fetch_html()
    snapshot = build_snapshot(page, season, gameweek=gameweek, deadline_time=deadline)

    # Fail CLOSED on an empty parse. "A partial snapshot beats no snapshot" is true of
    # 18/20 teams; it is false of 0/20, which is not data but a failed fetch wearing
    # data's clothes — a bot challenge or consent wall still answers 200. Archiving it
    # would poison an unrecoverable dataset with a file that *looks* like a record, and
    # (worse) do so silently. Refuse to save, and make the caller fail.
    if snapshot["team_count"] == 0:
        raise LineupFetchError("parsed 0 teams — not archiving. %s" % diagnose(page))
    if snapshot["team_count"] < EXPECTED_TEAMS:
        # A genuine partial IS worth keeping: it is real data, and the counts are
        # recorded so a later re-parse can spot it. Callers still flag it.
        logger.warning("parsed only %d/%d teams — the page layout may have changed",
                       snapshot["team_count"], EXPECTED_TEAMS)
    health = snapshot["health"]
    if health["degraded"]:
        # Teams parsed, but the median one lost its starting eleven: structural drift
        # in the table, not a light news day. Save it (real data, recoverable context)
        # but make sure the caller fails so it cannot rot the archive unnoticed.
        logger.warning("DEGRADED parse: median %d players/team (< %d) — the table "
                       "structure may have changed; thin teams: %s",
                       health["median_players_per_team"], MIN_PLAYERS_PER_TEAM,
                       ", ".join(health["thin_teams"]) or "none")
    elif health["thin_teams"]:
        logger.warning("%d team(s) parsed thin (< %d players): %s",
                       len(health["thin_teams"]), MIN_PLAYERS_PER_TEAM,
                       ", ".join(health["thin_teams"]))
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

    try:
        snapshot = archive_now(season=args.season, gameweek=args.gameweek)
    except LineupFetchError as exc:
        print("ERROR: %s" % exc)
        return 1
    print("archived %s GW=%s — %d teams, %d players, median %d/team (%s)" % (
        snapshot["season"], snapshot["gameweek"], snapshot["team_count"],
        snapshot["player_count"], snapshot["health"]["median_players_per_team"],
        snapshot["fetched_at"]))
    # A partial parse still exits non-zero so CI goes red: the data is saved (and
    # committable), but silence would let a degraded feed rot the archive unnoticed.
    # Two ways to be partial — too few teams, or teams too thin — and both count.
    ok = (snapshot["team_count"] >= EXPECTED_TEAMS
          and not snapshot["health"]["degraded"])
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
