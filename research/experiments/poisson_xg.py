"""Expected-Goals (xG) Poisson model - the Phase 1 hypothesis.

Identical machinery to the independent Poisson model (Maher, 1982), with ONE
change: team attack/defense strengths and the home advantage are estimated
from **Expected Goals** (xG) instead of actual goals scored. The prediction
target is unchanged - it still emits a full distribution over *actual* integer
scorelines, and hence P(H)/P(D)/P(A) - so it is directly comparable to the
goal-based models on the same matches.

Why this could win: actual goals are a noisy realisation of a team's true
chance-creation. A single deflection or worldie moves a scoreline but not a
team's underlying quality. xG measures the quality of chances created and
conceded, so strengths fitted on xG should be a less noisy estimate of how
good a team really is - and therefore a better predictor of future results.
This is the single most-cited reason xG is used in football analytics; Phase 1
puts that claim through the same walk-forward backtest every other model faces.

Fitting note: actual goals are counts, so the goal Poisson model maximises the
integer Poisson log-likelihood. xG is continuous and non-negative, so the
integer PMF does not apply. We instead minimise the **Poisson deviance**, the
standard quasi-likelihood for continuous non-negative targets:

    sum over matches of [ lambda - y * log(lambda) ]     (y = observed xG)

The dropped `log(y!)` term is constant in the parameters, so this yields
exactly the same estimating equations as Poisson MLE would if y happened to be
integer - keeping the estimator principled and, per this project's
"explainability before black-box AI" rule, every fitted number still a named,
inspectable attack/defense/home-advantage parameter.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.experiments.poisson import PoissonModel, _team_index


class PoissonXGModel(PoissonModel):
    """Poisson goals model whose strengths are fitted on xG, not goals.

    Inherits `score_grid` / `predict_proba` unchanged from `PoissonModel`:
    once attack/defense/home_advantage are fitted, predicting the actual-goal
    scoreline distribution is identical to the goal-based model.
    """

    def fit(self, train_df: pd.DataFrame) -> "PoissonXGModel":
        self._team_idx = _team_index(train_df)
        n_teams = len(self._team_idx)
        home_idx = train_df["home_team"].map(self._team_idx).to_numpy()
        away_idx = train_df["away_team"].map(self._team_idx).to_numpy()
        home_xg = train_df["home_xg"].to_numpy()
        away_xg = train_df["away_xg"].to_numpy()

        # Reference team (index 0) pinned at attack=defense=0 to resolve the
        # additive identifiability of the attack/defense parameters - same
        # convention as the goal-based Poisson model.
        n_free = n_teams - 1

        def unpack(params):
            attack = np.concatenate([[0.0], params[:n_free]])
            defense = np.concatenate([[0.0], params[n_free : 2 * n_free]])
            home_advantage = params[-1]
            return attack, defense, home_advantage

        def neg_poisson_deviance(params):
            attack, defense, home_advantage = unpack(params)
            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])
            # Poisson quasi-negative-log-likelihood for continuous targets:
            # sum(lambda - y*log(lambda)); minimised at the same parameters as
            # Poisson MLE. log(lambda) is finite since lambda = exp(...) > 0.
            return float(
                np.sum(lambda_home - home_xg * np.log(lambda_home))
                + np.sum(lambda_away - away_xg * np.log(lambda_away))
            )

        x0 = np.zeros(2 * n_free + 1)
        result = minimize(neg_poisson_deviance, x0=x0, method="L-BFGS-B")
        self._attack, self._defense, self._home_advantage = unpack(result.x)
        return self
