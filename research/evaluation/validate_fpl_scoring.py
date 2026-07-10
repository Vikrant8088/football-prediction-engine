"""Validate our FPL scoring rules against thousands of REAL scored matches.

Every fantasy projection is a weighted sum of the scoring rules, so a single
wrong constant silently corrupts every downstream number. Rather than trust the
rules as written, we check them: for a sample of real players, fetch FPL's own
per-match history, recompute each match's points with `scoring.match_points()`,
and compare against the `total_points` FPL actually awarded.

If our rules are right, the reconstruction is EXACT on essentially every match.
Any systematic mismatch is a rule we have misunderstood, and the report says so
with examples rather than burying it.

The FPL API is free and unauthenticated, but one request per player is needed,
so a polite delay is used and the sample is capped.
"""

import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.scoring import POSITION_NAMES, match_points_from_history
from research.data.fpl_loader import load_players
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

API = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
PER_POSITION = 15          # sample this many players per position
MIN_MINUTES = 900          # only players with a real body of matches
REQUEST_DELAY_SECONDS = 0.4


def _history(player_id: int) -> list:
    response = requests.get(f"{API}/element-summary/{player_id}/", headers=HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return response.json()["history"]


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="validate_fpl_scoring.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    players = load_players()
    eligible = players[players["minutes"] >= MIN_MINUTES]

    sample = (
        eligible.sort_values("total_points", ascending=False)
        .groupby("position")
        .head(PER_POSITION)
    )
    logger.info("Validating on %d players", len(sample))

    checked = 0
    exact = 0
    mismatches = []
    diff_counter = Counter()

    for player in sample.itertuples():
        for row in _history(player.id):
            if int(row["minutes"]) <= 0:
                continue  # unused sub: FPL awards 0, nothing to check
            predicted = match_points_from_history(player.position, row)
            actual = int(row["total_points"])
            checked += 1
            if predicted == actual:
                exact += 1
            else:
                diff_counter[predicted - actual] += 1
                if len(mismatches) < 12:
                    mismatches.append({
                        "player": player.web_name,
                        "position": POSITION_NAMES[player.position],
                        "gameweek": row["round"],
                        "predicted": predicted,
                        "actual": actual,
                        "minutes": row["minutes"],
                        "goals": row["goals_scored"],
                        "assists": row["assists"],
                        "clean_sheet": row["clean_sheets"],
                        "conceded": row["goals_conceded"],
                        "bonus": row["bonus"],
                        "defensive_contribution": row.get("defensive_contribution"),
                    })

    accuracy = exact / checked if checked else float("nan")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        f"# FPL Scoring-Rule Validation - {run_id}",
        "",
        f"Recomputed the points for **{checked} real scored matches** across "
        f"{len(sample)} players (top {PER_POSITION} per position with "
        f"{MIN_MINUTES}+ minutes), and compared with the points FPL actually "
        f"awarded.",
        "",
        "## Result",
        "",
        f"| | |",
        f"|---|---|",
        f"| Matches checked | {checked} |",
        f"| Reconstructed exactly | **{exact}** |",
        f"| **Exact-match rate** | **{accuracy:.4%}** |",
        f"| Mismatches | {checked - exact} |",
        "",
    ]
    if exact == checked:
        lines.append(
            "**Every single match reconstructs exactly.** The scoring rules in "
            "`prediction_engine/fpl/scoring.py` are correct, including the new "
            "2025/26 defensive-contribution rule. Every projection built on them "
            "inherits that correctness."
        )
    else:
        lines += [
            "**Some matches do not reconstruct.** Distribution of "
            "(predicted - actual):",
            "",
            "| difference | count |",
            "|---|---|",
        ]
        for diff, count in sorted(diff_counter.items()):
            lines.append(f"| {diff:+d} | {count} |")
        lines += ["", "### Example mismatches", "", "```",
                  json.dumps(mismatches, indent=2), "```"]

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fpl_scoring_validation_{run_id}.md").write_text(report, encoding="utf-8")
    print(report)
    return {"checked": checked, "exact": exact, "accuracy": accuracy}


if __name__ == "__main__":
    main()
