"""Independent Poisson goals model (Maher, 1982).

home_goals ~ Poisson(lambda_home), away_goals ~ Poisson(lambda_away),
independently, with:

    lambda_home = exp(attack[home_team] + defense[away_team] + home_advantage)
    lambda_away = exp(attack[away_team] + defense[home_team])

`attack[t]` is team t's log-scale scoring strength; `defense[t]` is team t's
log-scale leakiness (higher = concedes more). Fitted once per training
window via maximum likelihood (scipy.optimize, not a black-box GLM library -
consistent with this project's "explainability before black-box AI"
principle: every parameter here is a named, inspectable number).

Fixed columns/scoreline probabilities are derived from lambda_home/
lambda_away, never predicted directly - see the ML-framing discussion this
research milestone follows from: an outcome distribution is generated from
one small set of team-strength parameters, not learned as its own target.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from research.experiments.base import PredictionModel

MAX_GOALS = 10  # scoreline grid truncation; Poisson tail beyond this is negligible for football


def _team_index(df: pd.DataFrame) -> Dict[str, int]:
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return {team: i for i, team in enumerate(teams)}


class PoissonModel(PredictionModel):
    def __init__(self):
        self._team_idx: Dict[str, int] = {}
        self._attack: np.ndarray = None
        self._defense: np.ndarray = None
        self._home_advantage: float = 0.0

    def _lambdas(self, home_team: str, away_team: str) -> Tuple[float, float]:
        # Unseen teams (e.g. newly promoted, not present in the training
        # window) fall back to league-average strength (attack=defense=0)
        # rather than raising - a documented cold-start approximation.
        a_home = self._attack[self._team_idx[home_team]] if home_team in self._team_idx else 0.0
        d_home = self._defense[self._team_idx[home_team]] if home_team in self._team_idx else 0.0
        a_away = self._attack[self._team_idx[away_team]] if away_team in self._team_idx else 0.0
        d_away = self._defense[self._team_idx[away_team]] if away_team in self._team_idx else 0.0

        lambda_home = np.exp(a_home + d_away + self._home_advantage)
        lambda_away = np.exp(a_away + d_home)
        return lambda_home, lambda_away

    def fit(self, train_df: pd.DataFrame) -> "PoissonModel":
        self._team_idx = _team_index(train_df)
        n_teams = len(self._team_idx)
        home_idx = train_df["home_team"].map(self._team_idx).to_numpy()
        away_idx = train_df["away_team"].map(self._team_idx).to_numpy()
        home_goals = train_df["home_goals"].to_numpy()
        away_goals = train_df["away_goals"].to_numpy()

        # Reference team (index 0) is fixed at attack=defense=0 to resolve
        # the additive identifiability of attack/defense parameters (adding
        # a constant to every attack and subtracting it from home_advantage
        # leaves the likelihood unchanged unless one team is pinned down).
        n_free = n_teams - 1

        def unpack(params):
            attack = np.concatenate([[0.0], params[:n_free]])
            defense = np.concatenate([[0.0], params[n_free : 2 * n_free]])
            home_advantage = params[-1]
            return attack, defense, home_advantage

        def neg_log_likelihood(params):
            attack, defense, home_advantage = unpack(params)
            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])
            return -(
                poisson_dist.logpmf(home_goals, lambda_home).sum()
                + poisson_dist.logpmf(away_goals, lambda_away).sum()
            )

        x0 = np.zeros(2 * n_free + 1)
        result = minimize(neg_log_likelihood, x0=x0, method="L-BFGS-B")
        self._attack, self._defense, self._home_advantage = unpack(result.x)
        return self

    def score_grid(self, home_team: str, away_team: str) -> np.ndarray:
        """Full P(home_goals=h, away_goals=a) grid, shape (MAX_GOALS+1, MAX_GOALS+1)."""
        lambda_home, lambda_away = self._lambdas(home_team, away_team)
        home_pmf = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_home)
        away_pmf = poisson_dist.pmf(np.arange(MAX_GOALS + 1), lambda_away)
        return np.outer(home_pmf, away_pmf)

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._attack is None:
            raise RuntimeError("fit() must be called before predict_proba()")

        rows = np.arange(MAX_GOALS + 1)
        home_wins_mask = rows[:, None] > rows[None, :]
        draws_mask = rows[:, None] == rows[None, :]
        away_wins_mask = rows[:, None] < rows[None, :]

        probs = np.empty((len(fixtures_df), 3))
        for i, row in enumerate(fixtures_df.itertuples()):
            grid = self.score_grid(row.home_team, row.away_team)
            grid = grid / grid.sum()  # truncation renormalization
            probs[i] = [grid[home_wins_mask].sum(), grid[draws_mask].sum(), grid[away_wins_mask].sum()]
        return probs
