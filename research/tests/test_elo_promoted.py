"""Tests for the Elo promoted-team prior (Phase 6e).

A team appearing in a season it was not in the previous season is dropped to
initial_rating - promoted_penalty at its first match of that season; the window's
first season is never penalised (no prior to compare against); and the default
penalty of 0 reproduces the shipped model exactly.
"""

import unittest

import pandas as pd

from research.experiments.elo import EloModel

_RESULTS = ["H", "D", "A"]


def _df():
    """Two seasons. Season 1: A,B,C. Season 2: A,B,NEW — so C was relegated and
    NEW promoted. A/B continue throughout."""
    rows, day = [], 0
    fixtures = [
        ("2019-20", [("A", "B"), ("B", "C"), ("C", "A"), ("A", "C"), ("B", "A"), ("C", "B")]),
        ("2020-21", [("A", "B"), ("B", "NEW"), ("NEW", "A"), ("A", "NEW"), ("B", "A"), ("NEW", "B")]),
    ]
    for season, games in fixtures:
        for k, (h, a) in enumerate(games):
            rows.append({
                "season": season, "home_team": h, "away_team": a,
                "result": _RESULTS[k % 3],
                "home_goals": 1, "away_goals": 1,
                "date": pd.Timestamp("2019-08-01") + pd.Timedelta(days=day),
            })
            day += 7
    return pd.DataFrame(rows)


class TestEloPromotedPrior(unittest.TestCase):
    def _reset_teams(self, penalty=100.0):
        targets = EloModel(promoted_penalty=penalty)._promoted_first_matches(_df())
        return [team for teams in targets.values() for team in teams]

    def test_promoted_team_is_reset(self):
        self.assertIn("NEW", self._reset_teams())

    def test_continuing_teams_not_reset(self):
        reset = self._reset_teams()
        self.assertNotIn("A", reset)
        self.assertNotIn("B", reset)

    def test_first_season_teams_never_penalised(self):
        # C plays only in the window's first season -> no prior season exists.
        self.assertNotIn("C", self._reset_teams())

    def test_promoted_team_starts_lower(self):
        df = _df()
        base = EloModel(promoted_penalty=0.0).fit(df)._ratings
        pen = EloModel(promoted_penalty=150.0).fit(df)._ratings
        # NEW's whole Elo history begins 150 lower, so it ends up rated below where
        # the penalty-free model puts it.
        self.assertLess(pen["NEW"], base["NEW"])

    def test_penalty_zero_is_backward_compatible(self):
        df = _df()
        self.assertEqual(EloModel(promoted_penalty=0.0).fit(df)._ratings,
                         EloModel().fit(df)._ratings)


if __name__ == "__main__":
    unittest.main()
