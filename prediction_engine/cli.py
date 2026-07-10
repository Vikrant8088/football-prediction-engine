"""Command-line entry point: predict a real fixture.

    python -m prediction_engine.cli --home Arsenal --away Chelsea
    python -m prediction_engine.cli --league La_liga --home "Real Madrid" --away Barcelona
    python -m prediction_engine.cli --home Arsenal --away Chelsea --threshold 0.5 --grid
    python -m prediction_engine.cli --league Serie_A --list-teams
"""

import argparse
import logging
import sys
from typing import List

from prediction_engine.confidence import DEFAULT_THRESHOLD, coverage_at
from prediction_engine.engine import PredictionEngine

LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prediction_engine",
        description="Predict a football fixture with the benchmarked champion model.",
    )
    parser.add_argument("--league", default="EPL", choices=LEAGUES)
    parser.add_argument("--home", help="Home team name")
    parser.add_argument("--away", help="Away team name")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Publish a call only above this confidence (default %(default)s)",
    )
    parser.add_argument("--grid", action="store_true", help="Print the scoreline grid")
    parser.add_argument("--list-teams", action="store_true", help="List known teams and exit")
    return parser


def _print_grid(prediction, max_goals: int = 5) -> None:
    grid = prediction.scoreline_grid
    print("\n  Scoreline probabilities (%), home down / away across:")
    header = "        " + "".join(f"{a:>7}" for a in range(max_goals + 1))
    print(header)
    for home_goals in range(max_goals + 1):
        cells = "".join(f"{100 * grid[home_goals, a]:7.1f}" for a in range(max_goals + 1))
        print(f"    {home_goals}  {cells}")
    covered = grid[: max_goals + 1, : max_goals + 1].sum()
    print(f"  (this window covers {covered:.1%} of all outcomes)")


def main(argv: List[str] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = _build_parser().parse_args(argv)

    engine = PredictionEngine.train(args.league)

    if args.list_teams:
        print(f"{len(engine.teams)} teams in {args.league}:")
        for team in engine.teams:
            print(f"  {team}")
        return 0

    if not args.home or not args.away:
        print("error: --home and --away are required (or use --list-teams)", file=sys.stderr)
        return 2

    try:
        prediction = engine.predict(args.home, args.away, threshold=args.threshold)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(prediction.summary())
    if args.grid:
        _print_grid(prediction)
    print(
        f"\n  At a {args.threshold:.0%} threshold the engine calls "
        f"~{coverage_at(args.threshold):.0%} of all matches."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
