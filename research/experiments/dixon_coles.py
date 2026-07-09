"""Dixon-Coles model (Dixon & Coles, 1997, "Modelling Association Football
Scores and Inefficiencies in the Football Betting Market").

Extends the independent Poisson model two ways:

1. Low-score correlation adjustment (tau): the independent-Poisson assumption
   understates the empirical correlation between home and away goals in
   low-scoring games specifically (0-0, 1-0, 0-1, 1-1 occur at different
   rates than pure independence predicts). A single extra parameter, rho,
   multiplicatively corrects exactly those four cells of the scoreline grid.
2. Time-decay weighting (xi): more recent matches are weighted more heavily
   in the fit via an exponential decay on match age, reflecting that team
   strength drifts over a season rather than staying fixed across the whole
   training window.

xi is fixed at the original paper's published value (0.0065 per day) rather
than fit from this data - tuning it is a legitimate follow-up experiment,
not done here, and is called out explicitly so it isn't mistaken for a
validated choice.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as poisson_dist

from research.experiments.poisson import MAX_GOALS, PoissonModel, _team_index

XI_DEFAULT = 0.0065  # Dixon & Coles (1997) published time-decay rate, per day


def _tau(home_goals, away_goals, lambda_home, lambda_away, rho):
    """Dixon-Coles low-score correction factor, applied only to the four
    cells where it's non-trivial; 1.0 everywhere else."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel(PoissonModel):
    def __init__(self, xi: float = XI_DEFAULT):
        super().__init__()
        self.xi = xi
        self._rho = 0.0

    def fit(self, train_df: pd.DataFrame) -> "DixonColesModel":
        self._team_idx = _team_index(train_df)
        n_teams = len(self._team_idx)
        home_idx = train_df["home_team"].map(self._team_idx).to_numpy()
        away_idx = train_df["away_team"].map(self._team_idx).to_numpy()
        home_goals = train_df["home_goals"].to_numpy()
        away_goals = train_df["away_goals"].to_numpy()

        most_recent_date = train_df["date"].max()
        days_ago = (most_recent_date - train_df["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days_ago)

        n_free = n_teams - 1

        def unpack(params):
            attack = np.concatenate([[0.0], params[:n_free]])
            defense = np.concatenate([[0.0], params[n_free : 2 * n_free]])
            home_advantage = params[-2]
            rho = params[-1]
            return attack, defense, home_advantage, rho

        def neg_log_likelihood(params):
            attack, defense, home_advantage, rho = unpack(params)
            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])

            base_ll = poisson_dist.logpmf(home_goals, lambda_home) + poisson_dist.logpmf(
                away_goals, lambda_away
            )

            tau_adjustment = np.ones_like(base_ll)
            low_score_mask = (home_goals <= 1) & (away_goals <= 1)
            idx = np.where(low_score_mask)[0]
            for i in idx:
                tau_adjustment[i] = _tau(
                    home_goals[i], away_goals[i], lambda_home[i], lambda_away[i], rho
                )
            # tau can be slightly negative for extreme rho during
            # optimization; clip to keep log() finite and let the
            # likelihood penalize such parameter regions naturally.
            log_tau = np.log(np.clip(tau_adjustment, 1e-10, None))

            return -np.sum(weights * (base_ll + log_tau))

        x0 = np.zeros(2 * n_free + 2)
        result = minimize(neg_log_likelihood, x0=x0, method="L-BFGS-B")
        attack, defense, home_advantage, rho = unpack(result.x)
        self._attack, self._defense, self._home_advantage, self._rho = (
            attack,
            defense,
            home_advantage,
            rho,
        )
        return self

    def score_grid(self, home_team: str, away_team: str) -> np.ndarray:
        grid = super().score_grid(home_team, away_team)
        lambda_home, lambda_away = self._lambdas(home_team, away_team)
        for h in range(2):
            for a in range(2):
                grid[h, a] *= _tau(h, a, lambda_home, lambda_away, self._rho)
        # tau can (rarely, for an extreme fitted rho combined with a
        # fixture's specific lambdas) push a cell slightly negative; clip
        # so predict_proba always sees a valid, renormalizable distribution.
        return np.clip(grid, 0.0, None)
