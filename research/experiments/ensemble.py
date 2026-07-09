"""Weighted ensemble of base prediction models - the first Phase 2 model.

Phase 1 left us with two strong models that are good in *different* ways: Elo
(rating dynamics) and Dixon-Coles-xG (a calibrated scoreline distribution from
the xG signal). Neither dominates the other on every metric. That is exactly
the condition under which blending beats the best single model - the ensemble
can lean on whichever model is right where the other is wrong.

This blends the base models' probability vectors with a single set of weights:

    p_ensemble = sum_i w_i * p_i        (w_i >= 0, sum_i w_i = 1)

A convex combination of rows that each sum to 1 is itself a valid probability
row, so the output needs no renormalization. Weights are found by minimising
log loss, parameterised through a softmax so the non-negativity and sum-to-one
constraints hold by construction (no constrained optimiser needed).

Walk-forward safety (no leakage) is the whole game here. `fit(train_df)` never
touches the season it will predict; it learns weights with an INNER temporal
split of the training data only:

    1. split train_df -> inner_train (older) + inner_val (most recent
       `val_seasons` seasons);
    2. fit each base model on inner_train, predict inner_val, and fit the
       weights on those out-of-sample base predictions;
    3. refit every base model on the FULL train_df for actual prediction.

So the weights are chosen on data the base models did not train on, and the
whole ensemble is chosen on data strictly before the evaluated season - the
same discipline every other model in this project is held to. Because it
implements the standard fit/predict_proba contract, it drops straight into the
existing walk-forward harness as just another model.

The learned weights are inspectable (`.weights_`) and logged on every fit -
the ensemble says exactly how much it trusts each base model, keeping it within
the project's "explainability before black-box" rule.
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from research.experiments.base import OUTCOMES, PredictionModel

logger = logging.getLogger(__name__)

_ONEHOT = {"H": [1.0, 0.0, 0.0], "D": [0.0, 1.0, 0.0], "A": [0.0, 0.0, 1.0]}


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _onehot(outcomes) -> np.ndarray:
    return np.array([_ONEHOT[o] for o in outcomes], dtype=float)


def _fit_log_loss_weights(stacked_probs: np.ndarray, outcomes) -> np.ndarray:
    """stacked_probs: (K, m, 3) base-model probabilities on a validation set.
    Returns length-K weights (>=0, sum 1) minimising blended log loss."""
    onehot = _onehot(outcomes)
    k = stacked_probs.shape[0]

    def neg_log_loss(z):
        w = _softmax(z)
        blended = np.tensordot(w, stacked_probs, axes=(0, 0))  # (m, 3)
        picked = np.sum(blended * onehot, axis=1)
        return -np.mean(np.log(np.clip(picked, 1e-12, 1.0)))

    result = minimize(neg_log_loss, x0=np.zeros(k), method="Nelder-Mead")
    return _softmax(result.x)


class EnsembleModel(PredictionModel):
    def __init__(self, base_builders: dict, val_seasons: int = 2):
        """`base_builders`: {name: zero-arg builder returning a PredictionModel}.
        `val_seasons`: how many of the most recent training seasons to hold out
        internally for weight-fitting."""
        self.base_builders = dict(base_builders)
        self.val_seasons = val_seasons
        self._models = None
        self.weights_ = None  # {name: weight}, set on fit

    def _fit_weights(self, train_df: pd.DataFrame) -> np.ndarray:
        names = list(self.base_builders)
        seasons = sorted(train_df["season"].unique())
        if len(seasons) <= self.val_seasons:
            # Not enough history to hold seasons out - fall back to equal
            # weights rather than fitting on data a base model also trained on.
            return np.full(len(names), 1.0 / len(names))

        val_seasons = set(seasons[-self.val_seasons :])
        inner_train = train_df[~train_df["season"].isin(val_seasons)]
        inner_val = train_df[train_df["season"].isin(val_seasons)]
        val_fixtures = inner_val[["home_team", "away_team"]]

        stacked = np.stack(
            [
                self.base_builders[name]().fit(inner_train).predict_proba(val_fixtures)
                for name in names
            ],
            axis=0,
        )
        return _fit_log_loss_weights(stacked, inner_val["result"].to_numpy())

    def fit(self, train_df: pd.DataFrame) -> "EnsembleModel":
        names = list(self.base_builders)
        weights = self._fit_weights(train_df)
        self.weights_ = dict(zip(names, weights))
        logger.info(
            "Ensemble weights: %s",
            ", ".join(f"{n}={w:.3f}" for n, w in self.weights_.items()),
        )
        # Refit every base model on the full training window for prediction.
        self._models = {
            name: self.base_builders[name]().fit(train_df) for name in names
        }
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._models is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        blended = sum(
            self.weights_[name] * self._models[name].predict_proba(fixtures_df)
            for name in self._models
        )
        # Guard against tiny float drift so every row sums to exactly 1.
        return blended / blended.sum(axis=1, keepdims=True)
