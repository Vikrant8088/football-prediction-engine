"""Project FPL points for a fixture.

    python -m prediction_engine.fpl.cli --home Arsenal --away Chelsea
    python -m prediction_engine.fpl.cli --home Liverpool --away Everton --top 20
    python -m prediction_engine.fpl.cli --home Arsenal --away Chelsea --explain Saka
"""

import argparse
import logging
import sys
from typing import Dict, List

from prediction_engine.engine import PredictionEngine
from prediction_engine.fpl.projection import project_fixture
from research.data.fpl_archive import ALL_SEASONS, load_gameweeks
from research.data.fpl_loader import load_players

logger = logging.getLogger(__name__)


def _recent_minutes_history() -> Dict[int, list]:
    """Per-player CURRENT-season minutes (chronological) for the recent-form minutes
    model, keyed by FPL player id. Double-gameweeks are summed to the gameweek total,
    matching how the model was proven. Returns {} on any failure so the projection
    degrades gracefully — at the opening weeks, to the code-joined last-season
    cold-start (`bank_it.last_season_minutes`).

    Must be the *current* season, not `ALL_SEASONS[-1]` (the most-recent INGESTED
    season). Those diverge at GW1 of a new season: the archive still ends at last
    season, whose element ids have since been REASSIGNED, so keying this season's
    lookups by last season's ids silently hands each player a different player's
    minutes (measured: 2026/27 Saka got a benchwarmer's 352 minutes instead of his
    own 2218). Loading the real current season instead returns {} until its data
    exists, which is exactly when the code-joined cold-start should take over.
    """
    from research.data.predicted_lineups import current_season
    try:
        frame = load_gameweeks(current_season())
    except Exception as exc:                      # data not ingested yet / off-season
        logger.info("no current-season minutes history yet (%s); opening weeks will "
                    "cold-start from last season", exc)
        return {}
    history = {}
    per_gw = frame.groupby(["player_id", "gameweek"])["minutes"].sum().reset_index()
    for player_id, rows in per_gw.groupby("player_id"):
        history[int(player_id)] = rows.sort_values("gameweek")["minutes"].tolist()
    return history


def _safe(text: str) -> str:
    """Player names carry accents (Gyökeres, Ødegaard) but a Windows console is
    often cp1252. Degrade gracefully instead of mangling or crashing."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prediction_engine.fpl",
        description="Expected Fantasy Premier League points for a fixture.",
    )
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--top", type=int, default=12, help="how many players to show")
    parser.add_argument("--explain", help="break down one player's projection")
    return parser


def main(argv: List[str] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = _build_parser().parse_args(argv)

    engine = PredictionEngine.train("EPL")
    players = load_players()
    minutes_history = _recent_minutes_history()

    try:
        table = project_fixture(engine, players, args.home, args.away,
                                minutes_history=minutes_history)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prediction = engine.predict(args.home, args.away)
    print()
    print(f"{args.home} vs {args.away}")
    print(f"  Win/Draw/Loss : {prediction.probabilities['home']:.0%} / "
          f"{prediction.probabilities['draw']:.0%} / {prediction.probabilities['away']:.0%}")
    home_cs = table[table["team"] == args.home]["clean_sheet_probability"].iloc[0]
    away_cs = table[table["team"] == args.away]["clean_sheet_probability"].iloc[0]
    print(f"  Clean sheet   : {args.home} {home_cs:.0%}  |  {args.away} {away_cs:.0%}")
    print()

    shown = table[table["available"]].head(args.top)
    print(f"  {'player':<16}{'team':<22}{'pos':<5}{'cost':>6}{'xPts':>7}{'xG':>7}{'xA':>7}")
    print("  " + "-" * 70)
    for row in shown.itertuples():
        print(_safe(
            f"  {row.player:<16}{row.team:<22}{row.position:<5}{row.price:>6.1f}"
            f"{row.expected_points:>7.2f}{row.expected_goals:>7.2f}{row.expected_assists:>7.2f}"
        ))

    unavailable = table[~table["available"]].head(4)
    if not unavailable.empty:
        print("\n  Flagged (injury/doubt), excluded above:")
        for row in unavailable.itertuples():
            print(_safe(f"    {row.player} ({row.team})"))

    if args.explain:
        match = table[table["player"].str.lower() == args.explain.lower()]
        if match.empty:
            print(f"\n  '{args.explain}' not found in this fixture", file=sys.stderr)
            return 1
        p = match.iloc[0]
        print(_safe(f"\n  Breakdown for {p['player']} ({p['position']}, {p['team']}):"))
        print(f"    expected minutes   {p['expected_minutes']:.0f}  "
              f"(appearance factor {p['appearance_factor']:.2f})")
        for part in ["appearance", "goals", "assists", "clean_sheet", "conceded",
                     "saves", "defensive", "bonus", "cards"]:
            if abs(p[part]) > 1e-9:
                print(f"    {part:<18} {p[part]:+.2f}")
        print(f"    {'TOTAL':<18} {p['expected_points']:+.2f}")
        print("\n    (appearance/goals/assists/clean_sheet/conceded/saves are modelled")
        print("     from the fixture; bonus/cards/defensive use the player's own rates)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
