"""Feature store (Phase 3): point-in-time-correct match features.

The single non-negotiable rule (the whole reason backtests mean anything): a
feature for a match kicking off at time T may use ONLY information available
before T. No feature may peek at the match it describes or any later match.
Every builder here computes a team's features from that team's *strictly
earlier* matches only, and the leakage tests assert exactly that.

Features produced (per match, for both the home and away side):

- form (ppg / gd): points-per-game and average goal difference over the team's
  last `form_window` matches - a cheap, model-agnostic strength/momentum proxy.
- rest_days: days since the team's previous match. Short rest = fatigue.
- congestion: how many matches the team played in the `congestion_days` before
  this one. A pile-up of fixtures (e.g. midweek European games) tires a squad.

Rest and congestion are the "tiredness" signal - the free half of Phase 3,
computable from match dates alone. (Injuries and confirmed line-ups, the paid
half, will become additional builders here once an API-Football key is
available; they plug into the same point-in-time contract.)

Cold start: a team's first match in the dataset has no history, so its
form/rest come out as NaN (congestion is 0 - genuinely zero prior matches).
Downstream models fill NaN with a training-set neutral value, so a
newly-appearing team simply starts from league-average, never crashes.
"""

import numpy as np
import pandas as pd

FORM_FEATURES = ["home_form_ppg", "away_form_ppg", "home_gd_form", "away_gd_form"]
TIREDNESS_FEATURES = [
    "home_rest_days",
    "away_rest_days",
    "home_congestion",
    "away_congestion",
]
ALL_FEATURES = FORM_FEATURES + TIREDNESS_FEATURES


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def add_match_features(
    matches: pd.DataFrame, form_window: int = 5, congestion_days: int = 14
) -> pd.DataFrame:
    """Return `matches` (chronologically sorted) with the feature columns added.

    Requires columns: date, home_team, away_team, home_goals, away_goals.
    Each match is processed in date order; a team's features are computed from
    its history *before* that match, and only then is the match appended to the
    team's history - so a match can never contribute to its own features.
    """
    df = matches.sort_values("date", kind="stable").reset_index(drop=True).copy()

    columns = ALL_FEATURES
    feats = {c: np.full(len(df), np.nan) for c in columns}
    history = {}  # team -> list of {"date", "points", "gd"}, chronological

    for i, row in enumerate(df.itertuples()):
        for side, team in (("home", row.home_team), ("away", row.away_team)):
            past = history.get(team)
            if past:
                feats[f"{side}_rest_days"][i] = (row.date - past[-1]["date"]).days
                window = past[-form_window:]
                feats[f"{side}_form_ppg"][i] = np.mean([e["points"] for e in window])
                feats[f"{side}_gd_form"][i] = np.mean([e["gd"] for e in window])
                cutoff = row.date - pd.Timedelta(days=congestion_days)
                feats[f"{side}_congestion"][i] = sum(
                    1 for e in past if e["date"] >= cutoff
                )
            else:
                # No prior matches: form/rest undefined (NaN); congestion is a
                # genuine zero, not missing data.
                feats[f"{side}_congestion"][i] = 0.0

        # Append the current match to each team's history AFTER reading features.
        hg, ag = row.home_goals, row.away_goals
        history.setdefault(row.home_team, []).append(
            {"date": row.date, "points": _points(hg, ag), "gd": hg - ag}
        )
        history.setdefault(row.away_team, []).append(
            {"date": row.date, "points": _points(ag, hg), "gd": ag - hg}
        )

    for c in columns:
        df[c] = feats[c]
    return df
