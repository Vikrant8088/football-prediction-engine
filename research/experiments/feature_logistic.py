"""Multinomial logistic regression on feature-store features.

This is the first model in the project that consumes an arbitrary table of
match features rather than raw goals/xG - the piece Phase 3 needs to test
whether new signals (tiredness now; injuries/line-ups later) actually improve
prediction. It is the deliberately-explainable cousin of a black-box ML model:
every feature gets a single, inspectable coefficient per outcome, and it is fit
by explicit maximum likelihood (scipy), in keeping with this project's
"explainability before black-box AI" rule.

P(H/D/A) is a softmax over a linear score per outcome:

    score_H = 0                         (H fixed as the reference class)
    score_D = b_D + w_D . x
    score_A = b_A + w_A . x
    P = softmax(score_H, score_D, score_A)

Which columns feed in is passed to the constructor, so the SAME model class
supports the controlled Phase 3 experiment: fit it on form-only features, then
on form+tiredness features, and compare - any out-of-sample gain is the
tiredness signal's contribution, nothing else.

Feature handling: features are standardised using training-set statistics
(so coefficients are comparable and the optimiser is well-conditioned), and
missing values (a cold-start team with no history) are filled with the
training mean - i.e. treated as league-average - using train stats only, so
prediction never peeks at the evaluation set. A small L2 penalty keeps the fit
stable when features are collinear.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.experiments.base import PredictionModel

CLASSES = ("H", "D", "A")
_L2 = 1e-4  # tiny ridge penalty for numerical stability on standardized features


class FeatureLogisticModel(PredictionModel):
    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self._coef = None  # (2, n_params): rows for D and A; H is the reference
        self._mean = None
        self._std = None

    def _design_matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=float)
        if fit:
            self._mean = np.nanmean(X, axis=0)
            std = np.nanstd(X, axis=0)
            std[std == 0] = 1.0  # guard constant columns
            self._std = std
        # Fill missing (cold-start) values with the training mean, then
        # standardize - both using train-only statistics (no leakage).
        nan_rows, nan_cols = np.where(np.isnan(X))
        X[nan_rows, nan_cols] = np.take(self._mean, nan_cols)
        X_std = (X - self._mean) / self._std
        intercept = np.ones((len(X_std), 1))
        return np.hstack([intercept, X_std])

    def fit(self, train_df: pd.DataFrame) -> "FeatureLogisticModel":
        X = self._design_matrix(train_df, fit=True)
        y = np.array([CLASSES.index(r) for r in train_df["result"]])
        n_params = X.shape[1]

        def neg_log_likelihood(flat):
            coef = flat.reshape(2, n_params)  # classes D, A
            # Logit of the reference class H is fixed at 0.
            scores = np.hstack([np.zeros((len(X), 1)), X.dot(coef.T)])  # (n, 3)
            scores -= scores.max(axis=1, keepdims=True)
            log_z = np.log(np.exp(scores).sum(axis=1))
            log_lik = scores[np.arange(len(X)), y] - log_z
            return -log_lik.sum() + _L2 * np.sum(flat * flat)

        result = minimize(
            neg_log_likelihood, x0=np.zeros(2 * n_params), method="L-BFGS-B"
        )
        self._coef = result.x.reshape(2, n_params)
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        X = self._design_matrix(fixtures_df, fit=False)
        scores = np.hstack([np.zeros((len(X), 1)), X.dot(self._coef.T)])
        scores -= scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)  # columns H, D, A
