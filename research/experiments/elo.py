"""Elo rating model, adapted for football's three-way (H/D/A) outcome.

Two separate fitted pieces, deliberately kept distinct:

1. Rating dynamics: the classic chess-style Elo update (actual score 1/0.5/0
   for win/draw/loss, expected score via a logistic function of the rating
   gap, plus a fixed home-advantage bonus added to the home team's effective
   rating). Ratings are *replayed from scratch* over the full training
   window on every `fit()` call rather than incrementally cached - for a few
   thousand matches this replay costs milliseconds, and it keeps this class
   stateless/re-fittable per walk-forward fold without any risk of a stale
   rating snapshot leaking across folds.

2. Outcome-probability mapping: rating differences alone don't tell you
   P(draw) - a two-team logistic score is not naturally a 3-way probability.
   Following Hvattum & Arntzen (2010, "Using ELO ratings for match result
   prediction in association football"), an ordered logistic regression
   (ordered Away < Draw < Home) of the actual match outcome on the pre-match
   Elo difference is fit on the training set, giving a proper 3-way
   probability for any future rating gap.

Note on evaluation granularity: within a walk-forward fold, ratings used for
prediction are frozen at the end of the training window (i.e. at the start
of the evaluated season) - they are not updated match-by-match *during*
evaluation. This matches the same season-level granularity used for the
Poisson/Dixon-Coles models, so all four models are compared on equal terms.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.experiments.base import PredictionModel

AWAY, DRAW, HOME = 0, 1, 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Clipping keeps exp() inside float64 range for extreme rating gaps. The
    # sigmoid saturates long before |x|=700, so this changes no representable
    # value - it only avoids a spurious overflow warning.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


def _fit_ordered_logit(elo_diff: np.ndarray, outcome_idx: np.ndarray):
    """Fit P(Away/Draw/Home) as an ordered logistic function of elo_diff.

    Parameterizes the two cutpoints as c1 and c1 + exp(log_gap) so c2 > c1
    is guaranteed by construction (an invalid c2 <= c1 would produce a
    negative P(draw), which is meaningless).
    """

    def neg_log_likelihood(params):
        beta, c1, log_gap = params
        c2 = c1 + np.exp(log_gap)
        p_away = _sigmoid(c1 - beta * elo_diff)
        p_away_or_draw = _sigmoid(c2 - beta * elo_diff)
        p_home = 1.0 - p_away_or_draw
        p_draw = p_away_or_draw - p_away

        eps = 1e-12
        log_lik = (
            np.log(np.clip(p_away[outcome_idx == AWAY], eps, 1.0)).sum()
            + np.log(np.clip(p_draw[outcome_idx == DRAW], eps, 1.0)).sum()
            + np.log(np.clip(p_home[outcome_idx == HOME], eps, 1.0)).sum()
        )
        return -log_lik

    result = minimize(
        neg_log_likelihood, x0=[0.005, -0.3, 0.0], method="Nelder-Mead"
    )
    beta, c1, log_gap = result.x
    return beta, c1, c1 + np.exp(log_gap)


class EloModel(PredictionModel):
    def __init__(
        self, k_factor: float = 20.0, home_advantage: float = 100.0, initial_rating: float = 1500.0
    ):
        # k_factor and home_advantage are literature-typical values for
        # football Elo (see clubelo.com / eloratings.net conventions), not
        # tuned against this dataset - tuning them is a natural follow-up
        # experiment, not done here.
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self._ratings = None
        self._logit_params = None  # (beta, c1, c2)

    def _replay(self, df: pd.DataFrame):
        ratings = defaultdict(lambda: self.initial_rating)
        elo_diffs = np.empty(len(df))
        outcome_idx = np.empty(len(df), dtype=int)

        for i, row in enumerate(df.itertuples()):
            r_home = ratings[row.home_team]
            r_away = ratings[row.away_team]
            diff = r_home + self.home_advantage - r_away
            elo_diffs[i] = diff

            if row.result == "H":
                actual_home, outcome_idx[i] = 1.0, HOME
            elif row.result == "D":
                actual_home, outcome_idx[i] = 0.5, DRAW
            else:
                actual_home, outcome_idx[i] = 0.0, AWAY

            expected_home = _sigmoid(diff / 400.0 * np.log(10))
            delta = self.k_factor * (actual_home - expected_home)
            ratings[row.home_team] = r_home + delta
            ratings[row.away_team] = r_away - delta

        return dict(ratings), elo_diffs, outcome_idx

    def fit(self, train_df: pd.DataFrame) -> "EloModel":
        self._ratings, elo_diffs, outcome_idx = self._replay(train_df)
        self._logit_params = _fit_ordered_logit(elo_diffs, outcome_idx)
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._ratings is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        beta, c1, c2 = self._logit_params

        diffs = np.array(
            [
                self._ratings.get(row.home_team, self.initial_rating)
                + self.home_advantage
                - self._ratings.get(row.away_team, self.initial_rating)
                for row in fixtures_df.itertuples()
            ]
        )
        p_away = _sigmoid(c1 - beta * diffs)
        p_away_or_draw = _sigmoid(c2 - beta * diffs)
        p_home = 1.0 - p_away_or_draw
        p_draw = p_away_or_draw - p_away
        return np.column_stack([p_home, p_draw, p_away])
