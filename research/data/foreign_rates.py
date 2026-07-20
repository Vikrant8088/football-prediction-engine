"""Cold-start a summer signing from his previous league, not from nothing.

The GW1 cold start (`bank_it.last_season_rates`) fills a player's per-90 rates from
his last PREMIER LEAGUE season. A player who has never played in the Premier League
has none, so his rates stay 0.00 and the squad optimiser cannot see him at all — the
one blind spot documented in `docs/05`. Every summer that silently hides exactly the
players a manager is most interested in.

Understat covers the other four big leagues, and all of them are already in the raw
lake (2014-2025). So a striker signed from La Liga does have measured xG; it is simply
not English.

**The shrinkage is measured, not guessed.** Using the 442 players in our own cache who
moved to the Premier League with 900+ prior foreign minutes, comparing their last two
foreign seasons against their first PL season:

    xG/90   foreign -> PL ratio 0.81,  r = 0.85 (r^2 = 0.71)
    xA/90   foreign -> PL ratio 0.84,  r = 0.76 (r^2 = 0.58)

Scaling a foreign rate by that ratio cut the error against a player's actual first-PL
season roughly in half versus assuming the league average (xG MAE 0.055 vs 0.115;
xA 0.039 vs 0.065). The ratio is stable from a 1-minute to an 1800-minute threshold
(0.80-0.85), so it is not an artifact of dropping players who moved and flopped.

Honest limits, none of them small:
  - Understat covers five leagues. A signing from the Eredivisie, Liga MX, the
    Championship or Saudi Arabia is still invisible.
  - One ratio is applied to all four leagues. La Liga and Ligue 1 almost certainly do
    not translate identically; per-league factors were not fitted because the sample
    is thin once split four ways.
  - Matching is by NAME (Understat publishes no FPL id), so it inherits every hazard
    the lineup resolver already handles, and refuses rather than guesses.
  - This is a Gate A result. It has never been through Gate B: nobody has shown it
    improves the actual squad decision. It is a cold start for players who would
    otherwise be invisible, not a proven edge.
"""

import glob
import json
import logging
import os
from typing import Dict, Optional

from data_warehouse.config.loader import load_config
from research.data.lineup_resolver import squash

logger = logging.getLogger(__name__)

# Understat's own league codes, minus the EPL (which `last_season_rates` covers).
FOREIGN_LEAGUES = ("La_liga", "Bundesliga", "Serie_A", "Ligue_1")

# Measured on 442 real transfers — see the module docstring. Applied to a foreign
# per-90 rate to express it in Premier League terms.
XG_TRANSFER_RATIO = 0.81
XA_TRANSFER_RATIO = 0.84

# Below this a per-90 rate is noise: a striker with 200 minutes and one tap-in reads
# as 0.45 xG/90. Roughly ten full matches.
MIN_MINUTES = 900


def previous_start_year(today=None) -> int:
    """Understat's start-year code for the season just finished.

    Understat labels a season by the year it begins, so the 2025/26 season is 2025.
    In July 2026 the season just finished is therefore 2025.
    """
    from datetime import datetime, timezone
    today = today or datetime.now(timezone.utc)
    return today.year - 1 if today.month >= 7 else today.year - 2


def _season_file(league: str, start_year: int) -> Optional[str]:
    """Newest ingested snapshot for one league-season, or None."""
    root = load_config().raw_data_dir / "understat" / league / str(start_year)
    matches = sorted(glob.glob(os.path.join(str(root), "*", "%s_%d.json" % (league, start_year))))
    return matches[-1] if matches else None


def load_league_players(league: str, start_year: int) -> list:
    path = _season_file(league, start_year)
    if path is None:
        logger.warning("no ingested %s %d", league, start_year)
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("players", [])


def foreign_rates(start_year: int, leagues=FOREIGN_LEAGUES,
                  min_minutes: int = MIN_MINUTES) -> Dict[str, dict]:
    """{squashed player name: per-90 rates already scaled into PL terms}.

    Keyed by the squashed, accent-stripped name so it can be joined to FPL's own
    spelling — but the *identity* is Understat's player id, which is what makes the
    two lookalike cases separable:

      same id, two rows    a mid-season move between covered leagues. The spells are
                           SUMMED, so a player who played half a year in each is
                           judged on the whole season rather than half of it.
      two ids, one name    two different footballers. The name is dropped entirely.
                           Silently keeping the one with more minutes would hand one
                           player's finishing to another, and the optimiser would buy
                           him on it. A missing player merely stays invisible; a wrong
                           one is actively harmful.
    """
    # squashed name -> understat id -> accumulated totals
    by_name: Dict[str, Dict[str, dict]] = {}
    for league in leagues:
        for row in load_league_players(league, start_year):
            key = squash(row.get("player_name") or "")
            player_id = str(row.get("id") or "")
            if not key or not player_id:
                continue
            entry = by_name.setdefault(key, {}).setdefault(player_id, {
                "name": row.get("player_name"), "league": league,
                "team": row.get("team_title"), "minutes": 0.0,
                "xg": 0.0, "xa": 0.0, "cards": 0.0,
            })
            minutes = float(row.get("time") or 0)
            if minutes > entry["minutes"]:      # label by his main spell
                entry["team"], entry["league"] = row.get("team_title"), league
            entry["minutes"] += minutes
            entry["xg"] += float(row.get("xG") or 0)
            entry["xa"] += float(row.get("xA") or 0)
            entry["cards"] += (float(row.get("yellow_cards") or 0)
                               + 3.0 * float(row.get("red_cards") or 0))

    rates, ambiguous = {}, 0
    for key, people in by_name.items():
        if len(people) > 1:
            ambiguous += 1
            logger.info("dropping %r: %d different players share the name", key, len(people))
            continue
        entry = list(people.values())[0]
        if entry["minutes"] < min_minutes:
            continue
        per_90 = entry["minutes"] / 90.0
        rates[key] = {
            "name": entry["name"],
            "league": entry["league"],
            "team": entry["team"],
            "minutes": entry["minutes"],
            # Scaled into Premier League terms by the measured transfer ratios.
            "xg_per_90": (entry["xg"] / per_90) * XG_TRANSFER_RATIO,
            "xa_per_90": (entry["xa"] / per_90) * XA_TRANSFER_RATIO,
            "cards_per_90": entry["cards"] / per_90,
            # Understat measures none of these; the projection must not invent them.
            "saves_per_90": 0.0,
            "bonus_per_90": 0.0,
            "dc_per_90": 0.0,
        }
    logger.info("foreign rates: %d players over %d minutes from %s %d "
                "(%d names dropped as ambiguous)",
                len(rates), min_minutes, "/".join(leagues), start_year, ambiguous)
    return rates


def match_players(players, rates: Dict[str, dict],
                  only_missing_from: Dict[int, dict] = None) -> Dict[int, dict]:
    """{FPL player id: foreign rates}, matched by name.

    `only_missing_from` is the Premier League cold-start map; anyone already in it is
    skipped, so a domestic player's own PL history always wins over a foreign proxy.

    Names shared by two different footballers are already dropped by `foreign_rates`,
    so anything reaching here identifies exactly one player. A missing rate merely
    leaves him invisible; a wrong one would hand another player's finishing to him and
    the optimiser would buy him on it.
    """
    only_missing_from = only_missing_from or {}

    matched = {}
    for _, player in players.iterrows():
        player_id = int(player["id"])
        if player_id in only_missing_from:
            continue                       # he has real Premier League history
        if float(player.get("minutes") or 0) > 0:
            continue                       # he has played this season already
        for candidate in (player.get("full_name"), player.get("web_name")):
            row = rates.get(squash(candidate or ""))
            if row is not None:
                matched[player_id] = row
                break

    logger.info("foreign cold start: matched %d players", len(matched))
    return matched
