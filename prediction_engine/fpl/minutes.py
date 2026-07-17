"""Expected-minutes models for the FPL projection.

Minutes are the master switch of FPL scoring: a benched player scores 0, a
substitute who plays <60 min earns 1 appearance point and no clean sheet, a
starter earns 2 and is clean-sheet eligible. So *how long will he be on the
pitch* gates every other point a projection assigns.

The shipped model is a flat season average - total minutes / gameweeks played -
which has two known weaknesses:

  1. It is blind to RECENT change. A player who just lost his place (recent
     zeros) keeps a healthy projection off his early-season starts; a player who
     just broke into the team is under-rated by his slow start.
  2. It SMEARS the 90-or-0 reality. A player who alternates 90/0/90/0 averages
     45, which the projection reads as "plays ~75% of 60 minutes" - when really
     he starts about half the time and plays ~90 when he does. The two are very
     different for clean-sheet points (which need a genuine 60+ appearance).

This module gives every minutes model the same explicit interface, returning the
three quantities the projection actually needs:

    expected_minutes  the mean minutes we expect (scales goal/assist/save rates)
    p_60              P(plays 60+ min): clean-sheet eligibility, and the long
                      half of appearance points
    p_play            P(plays at all): the short (1-point) appearance half

`crude_minutes` reproduces the shipped model exactly (p_60 == p_play, both the
average divided by 60), so switching the projection onto this interface changes
nothing until a better model is passed. `recent_form_minutes` is the candidate:
recency-weighted, and it estimates p_60 / p_play from the actual fraction of
recent matches cleared, rather than inferring them from a single average.
"""

FULL_MATCH_MINUTES = 90
MINUTES_FOR_LONG_APPEARANCE = 60

# The half-life (in matches) proven on the FPL edge in Phase 6b: recent-form at
# this setting beat the shipped flat average by +2.95 pts/GW head-to-head
# (significant on both tests, 7/8 seasons). Half-life 1 was twitchier and failed
# Wilcoxon; 2 is the champion. This is the system's default minutes model.
DEFAULT_HALF_LIFE = 2.0

# Shape of a Premier League appearance, MEASURED on 2025/26 rather than guessed
# (29,757 player-fixtures; 11,498 appearances; 8,360 starts from the FPL bootstrap):
#
#   P_60_GIVEN_START  7,815 player-fixtures reached 60+ / 8,360 starts = 0.935.
#                     A starter usually completes the hour. Slightly optimistic, since
#                     a substitute introduced before the half-hour also reaches 60 and
#                     is counted here; the archive carries no `starts` column to
#                     separate them.
#   MEAN_MINUTES_WHEN_LONG   E[minutes | reached 60]  = 85.3
#   MEAN_MINUTES_WHEN_CAMEO  E[minutes | 1-59]        = 22.2
#
# These only matter when a predicted-lineup Start % is supplied (`lineup_minutes`).
P_60_GIVEN_START = 0.935
MEAN_MINUTES_WHEN_LONG = 85.3
MEAN_MINUTES_WHEN_CAMEO = 22.2


def _clip(value, low, high):
    return max(low, min(high, value))


def crude_minutes(total_minutes, matches, availability=1.0):
    """The shipped model: flat average minutes per match, scaled by availability.

    Returns p_60 == p_play == min(1, mean/60) so that, fed through the
    projection's appearance formula, it reproduces the historical `2 * p_60`
    exactly. This is the backward-compatible default, not an improvement.
    """
    mean = _clip((total_minutes / max(matches, 1)) * availability, 0.0, FULL_MATCH_MINUTES)
    p = min(1.0, mean / MINUTES_FOR_LONG_APPEARANCE)
    return {"expected_minutes": mean, "p_60": p, "p_play": p}


def recent_form_minutes(minutes_sequence, half_life_matches=4.0, availability=1.0):
    """Recency-weighted minutes from a player's per-match minutes history.

    `minutes_sequence` is chronological (most recent LAST) and includes the
    zeros of matches he did not play - those zeros are the rotation/injury signal
    the flat average throws away. Weights halve every `half_life_matches` matches
    into the past, so recent form dominates without discarding older evidence.

    Returns None for an empty history (nothing to go on - the caller should fall
    back to the crude prior or skip the player).
    """
    seq = [float(m) for m in minutes_sequence]
    if not seq:
        return None

    n = len(seq)
    # Most recent match (index n-1) has age 0 -> weight 1; older matches decay.
    weights = [0.5 ** ((n - 1 - i) / half_life_matches) for i in range(n)]
    wsum = sum(weights)

    exp_min = sum(w * m for w, m in zip(weights, seq)) / wsum
    p_60 = sum(w for w, m in zip(weights, seq) if m >= MINUTES_FOR_LONG_APPEARANCE) / wsum
    p_play = sum(w for w, m in zip(weights, seq) if m > 0.0) / wsum

    return {
        "expected_minutes": _clip(exp_min * availability, 0.0, FULL_MATCH_MINUTES),
        "p_60": _clip(p_60 * availability, 0.0, 1.0),
        "p_play": _clip(p_play * availability, 0.0, 1.0),
    }


def lineup_minutes(start_pct, recent=None, availability=1.0):
    """Minutes from a predicted-lineup **Start %**, the freshest rotation signal there
    is (Phase 6f: perfect minutes knowledge is worth ~+4.6 pts/GW over recent form).

    `start_pct` is P(the manager picks him), 0-100, from a pre-deadline predicted
    lineup. It is NOT P(60+) and NOT P(plays): a starter is occasionally hooked before
    the hour, and a non-starter can still come off the bench. So the three quantities
    the projection needs are rebuilt from it explicitly:

        p_60   = P(start) x P(reaches 60 | start)
        p_play = P(start) + P(no start) x P(cameo | no start)
        E[min] = p_60 x E[min | 60+] + P(1-59) x E[min | cameo]

    P(cameo | no start) comes from the player's OWN recent history (`recent`), not a
    league constant: a fringe forward who always gets 20 minutes is a different animal
    from a reserve keeper who never moves. Without history it is 0 — if he does not
    start we assume he does not feature, which is the conservative read.

    `availability` (FPL's `chance_of_playing`) still applies on top: the feed can be
    stale, and FPL's own flag is the official word on a player being unavailable.

    Known imprecision, stated rather than hidden: the 1-59 minute bucket lumps a
    starter hooked on the hour together with a late substitute, so both are valued at
    MEAN_MINUTES_WHEN_CAMEO (~22). The archive carries no `starts` column to separate
    them. It costs ~1.5 minutes on a nailed starter (~2%), which is well inside the
    noise of everything downstream.

    Returns the same {expected_minutes, p_60, p_play} contract as every other model
    here, so the projection is indifferent to which one produced it.
    """
    p_start = _clip((float(start_pct) / 100.0) * availability, 0.0, 1.0)
    p_60 = _clip(p_start * P_60_GIVEN_START, 0.0, 1.0)

    p_cameo_given_no_start = 0.0
    if recent is not None:
        p_60_hist = float(recent["p_60"])
        p_play_hist = float(recent["p_play"])
        if p_60_hist < 1.0:
            p_cameo_given_no_start = _clip(
                (p_play_hist - p_60_hist) / (1.0 - p_60_hist), 0.0, 1.0)

    p_play = _clip(p_start + (1.0 - p_start) * p_cameo_given_no_start, 0.0, 1.0)
    expected = (p_60 * MEAN_MINUTES_WHEN_LONG
                + max(0.0, p_play - p_60) * MEAN_MINUTES_WHEN_CAMEO)
    return {
        "expected_minutes": _clip(expected, 0.0, FULL_MATCH_MINUTES),
        "p_60": p_60,
        "p_play": p_play,
    }
