"""Turn the engine's fixture prediction into a ready-to-post match preview.

`prediction_engine.cli` already predicts any fixture in the five Understat leagues.
This wraps that in the two things a content account actually needs: the analyst
numbers (win/draw/loss, expected goals, likeliest scores, BTTS, over 2.5) and a
copy-paste social post carrying the same "our call, before kickoff" line the brand is
built on.

No new model — it reads the same scoreline grid the FPL engine uses. Give it a
fixture, get a post:

    python -m prediction_engine.preview --league laliga --home Barcelona --away "Real Madrid"
    python -m prediction_engine.preview --league seriea --list-teams
"""

import argparse
import logging
import sys
from typing import List, Optional

import numpy as np

from prediction_engine.engine import PredictionEngine

# Friendly aliases so nobody has to remember Understat's exact codes.
_LEAGUE_ALIASES = {
    "epl": "EPL", "pl": "EPL", "premierleague": "EPL", "premier": "EPL", "england": "EPL",
    "laliga": "La_liga", "la": "La_liga", "spain": "La_liga",
    "bundesliga": "Bundesliga", "germany": "Bundesliga", "bundes": "Bundesliga",
    "seriea": "Serie_A", "serie": "Serie_A", "italy": "Serie_A",
    "ligue1": "Ligue_1", "ligue": "Ligue_1", "france": "Ligue_1",
}
LEAGUE_LABELS = {"EPL": "Premier League", "La_liga": "La Liga",
                 "Bundesliga": "Bundesliga", "Serie_A": "Serie A", "Ligue_1": "Ligue 1"}
LEAGUE_HASHTAGS = {"EPL": "#PL", "La_liga": "#LaLiga", "Bundesliga": "#Bundesliga",
                   "Serie_A": "#SerieA", "Ligue_1": "#Ligue1"}


def resolve_league(name: str) -> str:
    """Map any reasonable spelling to the Understat code, else raise with the options."""
    key = "".join(ch for ch in name.lower() if ch.isalnum())
    if key in _LEAGUE_ALIASES:
        return _LEAGUE_ALIASES[key]
    if name in LEAGUE_LABELS:
        return name
    raise ValueError("unknown league %r; try one of: epl, laliga, bundesliga, seriea, ligue1"
                     % name)


def match_preview(engine: PredictionEngine, home: str, away: str) -> dict:
    """Every number a preview needs, read straight off the scoreline grid."""
    grid = engine.scoreline_grid(home, away)
    size = grid.shape[0]

    p_home = float(np.tril(grid, -1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, 1).sum())
    exp_home = float(sum(i * grid[i, :].sum() for i in range(size)))
    exp_away = float(sum(j * grid[:, j].sum() for j in range(size)))

    scorelines = sorted(((i, j, float(grid[i, j])) for i in range(size) for j in range(size)),
                        key=lambda t: -t[2])
    btts = float(grid[1:, 1:].sum())
    over_25 = float(sum(grid[i, j] for i in range(size) for j in range(size) if i + j >= 3))

    return {
        "home": home, "away": away,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "exp_home": exp_home, "exp_away": exp_away,
        "top_scores": [(i, j, p) for i, j, p in scorelines[:3]],
        "likeliest": scorelines[0][:2],
        "btts": btts, "over_25": over_25,
    }


def render_numbers(pv: dict) -> str:
    """The analyst read — for you, not the post."""
    i, j = pv["likeliest"]
    lines = [
        "%s vs %s" % (pv["home"], pv["away"]),
        "  %-14s %3.0f%%   Draw %3.0f%%   %-14s %3.0f%%" % (
            pv["home"], pv["p_home"] * 100, pv["p_draw"] * 100,
            pv["away"], pv["p_away"] * 100),
        "  Expected goals: %.1f - %.1f" % (pv["exp_home"], pv["exp_away"]),
        "  Likeliest scores: " + ", ".join("%d-%d (%.0f%%)" % (a, b, p * 100)
                                           for a, b, p in pv["top_scores"]),
        "  Both teams to score: %.0f%%   Over 2.5: %.0f%%" % (
            pv["btts"] * 100, pv["over_25"] * 100),
    ]
    return "\n".join(lines)


def render_post(pv: dict, league_code: str, handle: str = "@thelockerroomco") -> str:
    """The copy-paste social post, in the brand's before-kickoff voice."""
    label = LEAGUE_LABELS.get(league_code, league_code)
    tag = LEAGUE_HASHTAGS.get(league_code, "")
    # Favourite = the side (or draw) the model leans to, for the one-line verdict.
    outcomes = [(pv["p_home"], "%s win" % pv["home"]),
                (pv["p_draw"], "Draw"),
                (pv["p_away"], "%s win" % pv["away"])]
    fav_p, fav_label = max(outcomes)
    i, j = pv["likeliest"]
    return "\n".join([
        "\U0001F52E %s — The Locker Room model says:" % label.upper(),
        "",
        "%s  %.0f%%" % (pv["home"], pv["p_home"] * 100),
        "Draw       %.0f%%" % (pv["p_draw"] * 100),
        "%s  %.0f%%" % (pv["away"], pv["p_away"] * 100),
        "",
        "\U0001F4CA Expected goals: %.1f – %.1f" % (pv["exp_home"], pv["exp_away"]),
        "\U0001F3AF Most likely score: %d-%d" % (i, j),
        "⚽ Both teams to score: %.0f%%  |  Over 2.5: %.0f%%" % (
            pv["btts"] * 100, pv["over_25"] * 100),
        "",
        "Our call, locked in before kickoff. Agree? \U0001F447",
        "%s %s #TheLockerRoomCo" % (tag, "#FPL" if league_code == "EPL" else ""),
        handle,
    ])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prediction_engine.preview",
        description="Generate a ready-to-post match preview for any of the 5 leagues.")
    p.add_argument("--league", default="EPL", help="epl, laliga, bundesliga, seriea, ligue1")
    p.add_argument("--home", help="Home team")
    p.add_argument("--away", help="Away team")
    p.add_argument("--handle", default="@thelockerroomco", help="social handle for the post")
    p.add_argument("--list-teams", action="store_true", help="list known teams and exit")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # The post is emoji-rich (it's social copy); a cp1252 Windows console would crash
    # on it. Emit UTF-8 so the post comes out clean and copy-pasteable.
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = _build_parser().parse_args(argv)
    try:
        league = resolve_league(args.league)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    engine = PredictionEngine.train(league)

    if args.list_teams:
        print("%d teams in %s:" % (len(engine.teams), LEAGUE_LABELS.get(league, league)))
        for team in engine.teams:
            print("  %s" % team)
        return 0
    if not args.home or not args.away:
        print("error: --home and --away are required (or use --list-teams)", file=sys.stderr)
        return 2

    try:
        pv = match_preview(engine, args.home, args.away)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1

    print(render_numbers(pv))
    print("\n" + "-" * 48 + "\nREADY TO POST:\n")
    print(render_post(pv, league, handle=args.handle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
