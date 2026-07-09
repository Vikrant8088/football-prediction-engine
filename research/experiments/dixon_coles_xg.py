"""xG-aware Dixon-Coles model - the follow-up Phase 1 left open.

Phase 1 showed that fitting team strength on Expected Goals (xG) beats fitting
it on actual goals, but a plain xG-Poisson still lost to the Elo champion. The
open question was whether putting the better *input* (xG) inside a stronger
*structure* closes that gap. This model is that experiment: it takes the
xG-based strength estimation of PoissonXG and adds Dixon-Coles's two
innovations.

The two innovations are estimated from the data they each actually describe:

1. **Time decay (xi)** - recent matches weight more heavily, because team
   strength drifts. This applies naturally to the xG strength fit: every
   match's contribution is weighted by exp(-xi * days_ago).

2. **Low-score correlation (rho)** - the independent-Poisson assumption
   mis-states how often 0-0 / 1-0 / 0-1 / 1-1 occur. This correction is about
   *actual integer scorelines*, not xG, so it is fit on the real goals - after
   the strengths are fixed - rather than folded into the (continuous) xG fit
   where integer low-score cells have no meaning.

That gives a clean, two-stage, fully explainable fit:

    Stage 1: attack/defense/home_advantage   <- recency-weighted xG
    Stage 2: rho                             <- recency-weighted actual goals,
                                                strengths held fixed

Prediction (the actual-goal scoreline grid with the low-score tau correction)
is inherited unchanged from DixonColesModel: once the four parameters are
fitted, an xG-fitted Dixon-Coles predicts identically to a goal-fitted one.

xi is held at the Dixon-Coles (1997) published default, same as the goal-based
DixonColesModel, so this comparison isolates the goals-vs-xG input, not a
different decay rate. Tuning xi remains a pending experiment.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.experiments.dixon_coles import XI_DEFAULT, DixonColesModel, _tau
from research.experiments.poisson import _team_index


class DixonColesXGModel(DixonColesModel):
    def __init__(self, xi: float = XI_DEFAULT):
        super().__init__(xi=xi)

    def fit(self, train_df: pd.DataFrame) -> "DixonColesXGModel":
        self._team_idx = _team_index(train_df)
        n_teams = len(self._team_idx)
        home_idx = train_df["home_team"].map(self._team_idx).to_numpy()
        away_idx = train_df["away_team"].map(self._team_idx).to_numpy()
        home_xg = train_df["home_xg"].to_numpy()
        away_xg = train_df["away_xg"].to_numpy()
        home_goals = train_df["home_goals"].to_numpy()
        away_goals = train_df["away_goals"].to_numpy()

        most_recent_date = train_df["date"].max()
        days_ago = (most_recent_date - train_df["date"]).dt.days.to_numpy()
        weights = np.exp(-self.xi * days_ago)

        n_free = n_teams - 1

        def unpack(params):
            attack = np.concatenate([[0.0], params[:n_free]])
            defense = np.concatenate([[0.0], params[n_free : 2 * n_free]])
            home_advantage = params[-1]
            return attack, defense, home_advantage

        # --- Stage 1: team strengths from xG, recency-weighted -------------
        # Poisson deviance (the continuous-target quasi-likelihood used in
        # PoissonXG) with each match weighted by its exponential time decay.
        def neg_weighted_deviance(params):
            attack, defense, home_advantage = unpack(params)
            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])
            return float(
                np.sum(weights * (lambda_home - home_xg * np.log(lambda_home)))
                + np.sum(weights * (lambda_away - away_xg * np.log(lambda_away)))
            )

        x0 = np.zeros(2 * n_free + 1)
        result = minimize(neg_weighted_deviance, x0=x0, method="L-BFGS-B")
        attack, defense, home_advantage = unpack(result.x)
        self._attack, self._defense, self._home_advantage = attack, defense, home_advantage

        # --- Stage 2: low-score correlation rho from ACTUAL goals ----------
        # Strengths are fixed, so the base Poisson term is constant in rho and
        # drops out; only the tau correction on the low-score matches remains.
        lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_advantage)
        lambda_away = np.exp(attack[away_idx] + defense[home_idx])
        low_score_idx = np.where((home_goals <= 1) & (away_goals <= 1))[0]

        def neg_weighted_rho_ll(rho_arr):
            rho = rho_arr[0]
            total = 0.0
            for i in low_score_idx:
                tau = _tau(
                    home_goals[i], away_goals[i], lambda_home[i], lambda_away[i], rho
                )
                total += weights[i] * np.log(np.clip(tau, 1e-10, None))
            return -total

        rho_result = minimize(neg_weighted_rho_ll, x0=np.array([0.0]), method="L-BFGS-B")
        self._rho = float(rho_result.x[0])
        return self
