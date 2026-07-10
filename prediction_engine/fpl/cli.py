"""Project FPL points for a fixture.

    python -m prediction_engine.fpl.cli --home Arsenal --away Chelsea
    python -m prediction_engine.fpl.cli --home Liverpool --away Everton --top 20
    python -m prediction_engine.fpl.cli --home Arsenal --away Chelsea --explain Saka
"""

import argparse
import logging
import sys
from typing import List

from prediction_engine.engine import PredictionEngine
from prediction_engine.fpl.projection import project_fixture
from research.data.fpl_loader import load_players


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

    try:
        table = project_fixture(engine, players, args.home, args.away)
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
