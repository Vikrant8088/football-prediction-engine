"""Watch for FPL publishing the new season, because everything else waits on it.

Two pieces of work are blocked until the 2026/27 game opens, and both have a hard
deadline (GW1, Fri 21 Aug 2026, 18:30 BST):

    M3      re-check the season's SCORING and TRANSFER/CHIP rules against
            `scoring.py` and `manager.py`. A proof computed on stale rules is void,
            and FPL has changed rules in consecutive seasons — the free-transfer cap
            went from 2 to 5 in 2024/25, and 2025/26 introduced defensive
            contribution. Assuming they did not change again is exactly the kind of
            assumption this project does not get to make.
    refresh re-download bootstrap and fixtures, retrain the engine on the new
            promoted teams, re-ingest Understat.

Neither can start early and neither should start late, so the rollover date matters.
It is not announced in advance and drifts year to year, which makes it precisely the
sort of thing a human forgets to check and a daily job never does.

Detection is deliberately simple and hard to fool. Between seasons FPL serves the
FINISHED season: every event is `finished`, none is flagged `is_next`. The moment the
new game is published there is an unfinished event with a future deadline. So:

    rolled over  <=>  an unfinished event exists

The team list is reported alongside it, not used as the trigger. It is the useful
detail for a human (which clubs were promoted, and therefore which have no Understat
history and will be cold-started), but it changes at a slightly different moment from
the events, and a two-signal trigger is two things that can disagree.

stdlib + requests only, no pandas and no ingested lake, so this runs in CI on a clean
checkout — the same reason `predicted_lineups` is standalone.
"""

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
TIMEOUT = 30

# The clubs of the season we are waiting to LEAVE. Anything here that vanishes was
# relegated; anything new was promoted and has no Understat history, so it will be
# cold-started. Recorded as context for a human, never as the trigger.
KNOWN_TEAMS_2025_26 = frozenset({
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Burnley",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds", "Liverpool",
    "Man City", "Man Utd", "Newcastle", "Nott'm Forest", "Spurs", "Sunderland",
    "West Ham", "Wolves",
})


def fetch_bootstrap(url: str = FPL_BOOTSTRAP_URL) -> dict:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def season_status(payload: dict) -> dict:
    """Has FPL published a new season? Returns a JSON-serialisable verdict.

    `rolled_over` is the only field anything should branch on. The rest is detail for
    whoever reads the alert.
    """
    events = payload.get("events") or []
    teams = sorted(str(team["name"]) for team in (payload.get("teams") or []))

    unfinished = [e for e in events if not e.get("finished")]
    flagged_next = [e for e in events if e.get("is_next")]
    # Prefer FPL's own `is_next` when it is set; fall back to the first unfinished
    # event, which is what exists in the window after publication but before the flag
    # settles.
    upcoming = (flagged_next or unfinished or [None])[0]

    incoming = sorted(set(teams) - KNOWN_TEAMS_2025_26)
    departed = sorted(KNOWN_TEAMS_2025_26 - set(teams))

    return {
        "rolled_over": bool(unfinished),
        "events": len(events),
        "finished_events": sum(1 for e in events if e.get("finished")),
        "next_gameweek": (int(upcoming["id"]) if upcoming else None),
        "next_deadline": (upcoming.get("deadline_time") if upcoming else None),
        "players": len(payload.get("elements") or []),
        "teams": teams,
        # Promoted clubs have no Understat history and will be cold-started, so this
        # is the first thing to look at once the refresh runs.
        "promoted": incoming,
        "relegated": departed,
    }


def check(url: str = FPL_BOOTSTRAP_URL) -> dict:
    status = season_status(fetch_bootstrap(url))
    if status["rolled_over"]:
        logger.info("FPL has rolled over: GW%s, deadline %s, %d players, promoted %s",
                    status["next_gameweek"], status["next_deadline"],
                    status["players"], ", ".join(status["promoted"]) or "none detected")
    else:
        logger.info("still the finished season (%d/%d events finished); nothing to do",
                    status["finished_events"], status["events"])
    return status


def issue_title(status: dict) -> str:
    deadline = status.get("next_deadline") or "unknown"
    return "FPL is open — GW%s, deadline %s" % (status.get("next_gameweek"), deadline)


def issue_body(status: dict) -> str:
    """The alert a human actually reads, with the blocked work as a checklist.

    Built here rather than in the workflow's shell on purpose: it contains backticks,
    asterisks and em-dashes, and assembling it through YAML -> shell -> jq is three
    quoting layers that cannot be tested locally. Here it is a pure function with
    tests, and the workflow only has to write it to a file.
    """
    promoted = ", ".join(status.get("promoted") or []) or "none detected"
    return "\n".join([
        "The new season is published. Detected automatically by "
        "`research/data/fpl_season_watch.py`.",
        "",
        "- **Next gameweek:** GW%s" % status.get("next_gameweek"),
        "- **Deadline:** %s" % status.get("next_deadline"),
        "- **Players in the game:** %s" % status.get("players"),
        "- **Promoted (no Understat history — will be cold-started):** %s" % promoted,
        "",
        "## Blocked work that can now start",
        "",
        "- [ ] **M3 — re-check the rules.** Verify the new season's **scoring** "
        "against `prediction_engine/fpl/scoring.py` AND its **transfer/chip** rules "
        "against `prediction_engine/fpl/manager.py`. FPL changed the free-transfer "
        "cap (2 → 5) in 2024/25 and added defensive contribution in 2025/26, so "
        "assuming nothing changed is not available to us — a proof computed on stale "
        "rules is void. The golden tests in `test_golden_projection.py` will fail "
        "loudly if `scoring.py` changes; that failure is the *point*, and it needs "
        "the backtest rerun alongside it.",
        "- [ ] **Refresh the data.** Re-download bootstrap + fixtures, retrain the "
        "engine, re-ingest Understat.",
        "- [ ] **Check the promoted clubs** cold-start sensibly (no Understat "
        "history).",
        "- [ ] **Re-run the backtests** on refreshed data and confirm the numbers "
        "still hold.",
        "- [ ] **M5 — lock the first real squad** before the GW1 deadline, and commit "
        "it *before* the deadline passes.",
        "",
        "Close this issue once the checklist is done.",
    ])


def main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="directory to write status.json, issue_title.txt "
                                      "and issue_body.md into (for CI)")
    args = parser.parse_args(argv)

    status = check()
    print(json.dumps(status, indent=2))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if status["rolled_over"]:
            (out / "issue_title.txt").write_text(issue_title(status), encoding="utf-8")
            (out / "issue_body.md").write_text(issue_body(status), encoding="utf-8")

    # Exit code is the signal the workflow branches on: 0 = still waiting, 1 = open.
    # Deliberately NOT an exception — "the new season exists" is good news, and a
    # traceback would read as a broken job.
    return 1 if status["rolled_over"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
