"""Golden values: the model's numbers must not move by accident.

Every other test asks "does it work?". This asks "does it still give the SAME
ANSWER?" — a different question, and the one this project's rules actually turn on:

    Never skip benchmarking. Every new model must outperform the previous version
    before replacing it.                                    (CLAUDE.md)

That rule is unenforceable if the projection can drift silently. A refactor that
nudges expected points by 2%, a reordered term, a changed constant — none of it fails
a behavioural test, none of it is visible in review, and all of it invalidates the
backtested edge the whole system is built on.

So these freeze exact outputs for fixed inputs. They are deliberately brittle: that is
the feature, not a flaw.

**If one of these fails, do not simply update the number.** A failure means the
model's answers changed. Either it was an accident — in which case the fix is to
revert, not to re-bless — or it was deliberate, in which case the change needs a
backtest showing it beats the champion, and the new value is recorded together with
the run that justified it. Rewriting a golden value to make CI green is the exact
self-deception the project exists to avoid.

Nothing here touches the data lake, the network or a fitted model, so the values are
reproducible anywhere, forever.
"""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import poisson

from prediction_engine.fpl.minutes import (crude_minutes, lineup_minutes,
                                           recent_form_minutes)
from prediction_engine.fpl.optimizer import select_squad
from prediction_engine.fpl.projection import fixture_context, project_player
from prediction_engine.fpl.scoring import match_points

PLACES = 8

# A fixed fixture: home team ~1.6 expected goals, away ~1.1.
_H = poisson.pmf(np.arange(8), 1.6)
_A = poisson.pmf(np.arange(8), 1.1)
GRID = np.outer(_H, _A)
GRID = GRID / GRID.sum()

TEAM_RATE = {"scored_per_match": 1.55, "conceded_per_match": 1.20}
MINUTES = {"expected_minutes": 88.0, "p_60": 0.95, "p_play": 0.99}

ARCHETYPES = {
    "elite_striker":   dict(position=4, xg=0.72, xa=0.18, saves=0.0, bonus=0.55, dc=1.2, cards=0.10),
    "creative_mid":    dict(position=3, xg=0.28, xa=0.42, saves=0.0, bonus=0.40, dc=4.5, cards=0.15),
    "nailed_defender": dict(position=2, xg=0.06, xa=0.09, saves=0.0, bonus=0.22, dc=11.0, cards=0.22),
    "goalkeeper":      dict(position=1, xg=0.00, xa=0.01, saves=3.1, bonus=0.18, dc=0.0, cards=0.04),
}

# Frozen from the shipped model. Full breakdowns, not just totals: a compensating
# pair of errors could leave the total unchanged while both channels are wrong.
GOLDEN_PROJECTION = {
    "elite_striker": {
        "expected_points": 5.8281581407, "appearance": 1.94, "goals": 2.9037121104,
        "assists": 0.5444460207, "clean_sheet": 0.0, "conceded": 0.0, "saves": 0.0,
        "bonus": 0.5377777778, "defensive": 0.0000000097, "cards": -0.0977777778,
        "expected_goals": 0.7259280276, "expected_assists": 0.1814820069,
    },
    "creative_mid": {
        "expected_points": 5.1865942312, "appearance": 1.94, "goals": 1.4115267203,
        "assists": 1.2703740483, "clean_sheet": 0.3162338934, "conceded": 0.0,
        "saves": 0.0, "bonus": 0.3911111111, "defensive": 0.0040151248,
        "cards": -0.1466666667, "expected_goals": 0.2823053441,
        "expected_assists": 0.4234580161,
    },
    "nailed_defender": {
        "expected_points": 4.7845427283, "appearance": 1.94, "goals": 0.3629640138,
        "assists": 0.2722230103, "clean_sheet": 1.2649355735, "conceded": -0.3203460289,
        "saves": 0.0, "bonus": 0.2151111111, "defensive": 1.2647661595,
        "cards": -0.2151111111, "expected_goals": 0.0604940023,
        "expected_assists": 0.0907410034,
    },
    "goalkeeper": {
        "expected_points": 3.9777790690, "appearance": 1.94, "goals": 0.0,
        "assists": 0.0302470011, "clean_sheet": 1.2649355735, "conceded": -0.3203460289,
        "saves": 0.9260536343, "bonus": 0.176, "defensive": 0.0,
        "cards": -0.0391111111, "expected_goals": 0.0,
        "expected_assists": 0.0100823337,
    },
}

# Note these sit slightly BELOW the Poisson lambdas (1.6 / 1.1): the grid is truncated
# at 8 goals and renormalised, so a sliver of mass is lost. Assuming they would equal
# the lambdas is exactly the kind of plausible-but-wrong value this suite exists to
# catch — it caught mine.
GOLDEN_CONTEXT_HOME = {
    "expected_goals_for": 1.5982790380,
    "expected_goals_against": 1.0998584220,
    "clean_sheet_probability": 0.3328777825,
    "expected_conceded_penalty": 0.3276266205,
}

GOLDEN_MINUTES = {
    "crude": {"expected_minutes": 75.0, "p_60": 1.0, "p_play": 1.0},
    "recent_form": {"expected_minutes": 62.6684122383, "p_60": 0.5705255323,
                    "p_play": 0.8221058508},
    "lineup": {"expected_minutes": 60.898875, "p_60": 0.70125, "p_play": 0.75},
}

# Already validated to 100% on 2,085 real scored matches; frozen here so a refactor
# of the rules cannot quietly change them.
# Each derived by hand from the rules, then confirmed against the code — not copied
# from a run. (Writing these out caught an arithmetic slip of mine, which is the
# entire argument for deriving rather than blessing whatever the code emits.)
GOLDEN_SCORING = {
    # 2 appearance + 2 goals x4 (FWD) + 1 assist x3 + 3 bonus; forwards take no
    # goals-conceded penalty.
    "striker_brace": 16,
    # 2 appearance + 4 clean sheet (DEF) + 1 bonus - 1 yellow.
    "defender_clean_sheet": 6,
    # 2 appearance + 4 clean sheet (GKP) + 7//3 = 2 saves + 2 bonus.
    "keeper_seven_saves": 10,
    # 1 appearance (under 60 min) + 1 assist x3; midfielders take no conceded penalty.
    "sub_cameo": 4,
}


def _player(spec):
    return pd.Series({
        "position": spec["position"], "minutes": 900, "available": True,
        "chance_of_playing": 100.0,
        "xg_per_90": spec["xg"], "xa_per_90": spec["xa"],
        "saves_per_90": spec["saves"], "bonus_per_90": spec["bonus"],
        "dc_per_90": spec["dc"], "cards_per_90": spec["cards"],
    })


_ADVICE = ("\n\nThe model's numbers changed. If this was accidental, REVERT — do not "
           "re-bless the value. If deliberate, it needs a backtest showing it beats "
           "the champion, and the new value recorded with the run that justified it.")


class TestGoldenFixtureContext(unittest.TestCase):
    def test_context_read_off_the_grid_is_unchanged(self):
        context = fixture_context(GRID, is_home=True)
        for key, expected in GOLDEN_CONTEXT_HOME.items():
            self.assertAlmostEqual(context[key], expected, places=PLACES,
                                   msg="fixture_context[%r] moved.%s" % (key, _ADVICE))


class TestGoldenProjection(unittest.TestCase):
    def test_every_archetype_projects_exactly_as_before(self):
        context = fixture_context(GRID, is_home=True)
        for name, spec in ARCHETYPES.items():
            actual = project_player(_player(spec), context, TEAM_RATE,
                                    minutes_model=dict(MINUTES))
            for channel, expected in GOLDEN_PROJECTION[name].items():
                self.assertAlmostEqual(
                    actual[channel], expected, places=PLACES,
                    msg="%s.%s moved: %.10f != %.10f%s"
                        % (name, channel, actual[channel], expected, _ADVICE))

    def test_the_parts_still_sum_to_the_total(self):
        """Guards the other direction: a channel could move while the frozen total
        stays right only if another cancels it. This makes that impossible."""
        context = fixture_context(GRID, is_home=True)
        for name, spec in ARCHETYPES.items():
            out = project_player(_player(spec), context, TEAM_RATE,
                                 minutes_model=dict(MINUTES))
            parts = sum(out[c] for c in ("appearance", "goals", "assists", "clean_sheet",
                                         "conceded", "saves", "bonus", "defensive", "cards"))
            self.assertAlmostEqual(parts, out["expected_points"], places=PLACES,
                                   msg="%s: parts do not sum to the total" % name)


class TestGoldenMinutes(unittest.TestCase):
    def test_crude_is_unchanged(self):
        self._compare(crude_minutes(900, 12), GOLDEN_MINUTES["crude"], "crude")

    def test_recent_form_is_unchanged(self):
        self._compare(recent_form_minutes([90, 90, 0, 45, 90], half_life_matches=2.0),
                      GOLDEN_MINUTES["recent_form"], "recent_form")

    def test_lineup_start_pct_is_unchanged(self):
        recent = recent_form_minutes([90, 0, 90, 0], half_life_matches=2.0)
        self._compare(lineup_minutes(75, recent=recent),
                      GOLDEN_MINUTES["lineup"], "lineup")

    def _compare(self, actual, expected, label):
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value, places=PLACES,
                                   msg="%s.%s moved.%s" % (label, key, _ADVICE))


class TestGoldenScoringRules(unittest.TestCase):
    """The rules themselves. Changing one is legitimate only when FPL changes it —
    in which case this failing is the point (see M3 in docs/04)."""

    def test_scoring_rules_are_unchanged(self):
        cases = {
            "striker_brace": dict(position=4, minutes=90, goals_scored=2, assists=1,
                                  goals_conceded=1, bonus=3),
            "defender_clean_sheet": dict(position=2, minutes=90, clean_sheets=1,
                                         bonus=1, yellow_cards=1),
            "keeper_seven_saves": dict(position=1, minutes=90, clean_sheets=1,
                                       saves=7, bonus=2),
            "sub_cameo": dict(position=3, minutes=20, assists=1, goals_conceded=2),
        }
        for name, kwargs in cases.items():
            self.assertEqual(
                match_points(**kwargs), GOLDEN_SCORING[name],
                "scoring rule %r changed. Legitimate ONLY if FPL changed it — in "
                "which case update this and scoring.py together, and say so." % name)


class TestGoldenSquadSelection(unittest.TestCase):
    """The optimiser is proven exact elsewhere. This pins that, given identical
    projections, it keeps making the identical choice — tie-breaking included."""

    def _frame(self):
        rows, pid = [], 0
        for club in range(6):
            for position in (1, 1, 2, 2, 3, 3, 4, 4):
                pid += 1
                rows.append({"player_id": pid, "player": "P%d" % pid,
                             "position": position, "club": "C%d" % club,
                             "price": 4.0 + (pid % 8) * 0.4,
                             # Deterministic, finely spaced, no ties.
                             "value": round(1.0 + (pid * 37 % 91) / 13.0, 4)})
        return pd.DataFrame(rows)

    def test_the_same_projections_select_the_same_squad(self):
        # Frozen from the solver, but only after checking independently that the squad
        # it returns is legal: 15 players, quota 2/5/5/3, max 3 per club, inside
        # budget, captain the highest-value starter. `test_squad_obeys_every_fpl_rule`
        # in the integration suite re-checks those properties on every run.
        frame = self._frame()
        squad = select_squad(frame, "value", squad_budget=100.0)
        self.assertIsNotNone(squad)
        self.assertEqual(sorted(int(frame.loc[i, "player_id"]) for i in squad.xi),
                         [7, 12, 14, 17, 22, 24, 27, 29, 39, 44, 46],
                         "the optimiser picked a different XI from identical inputs."
                         + _ADVICE)
        self.assertEqual(int(frame.loc[squad.captain, "player_id"]), 27)
        self.assertEqual(squad.formation, (3, 4, 3))
        self.assertAlmostEqual(squad.cost, 84.4, places=6)


if __name__ == "__main__":
    unittest.main()
