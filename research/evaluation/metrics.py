"""Proper scoring rules for 3-way (H/D/A) football outcome predictions.

Per the earlier ML-framing discussion: exact-score hit rate is deliberately
NOT part of this module's headline metrics - it's reported separately, if at
all, as a footnote. These three (log loss, RPS, Brier) plus calibration are
what actually measure whether a model's probabilities are good, not just
whether its single most-likely pick happened to be right.

All functions take:
    probs: (n, 3) array, columns ordered (H, D, A), rows summing to 1
    outcomes: length-n array of "H"/"D"/"A" strings

and return a single float (mean over all n matches).
"""

import numpy as np

OUTCOME_ORDER = ("H", "D", "A")
_OUTCOME_TO_ONEHOT = {
    "H": np.array([1.0, 0.0, 0.0]),
    "D": np.array([0.0, 1.0, 0.0]),
    "A": np.array([0.0, 0.0, 1.0]),
}


def _onehot(outcomes) -> np.ndarray:
    return np.array([_OUTCOME_TO_ONEHOT[o] for o in outcomes])


def log_loss(probs: np.ndarray, outcomes) -> float:
    """Mean negative log-likelihood of the actual outcome under the
    predicted distribution. Lower is better; heavily penalizes confident
    wrong predictions."""
    onehot = _onehot(outcomes)
    eps = 1e-12
    picked_probs = np.sum(np.clip(probs, eps, 1.0) * onehot, axis=1)
    return float(-np.mean(np.log(picked_probs)))


def brier_score(probs: np.ndarray, outcomes) -> float:
    """Mean squared error between predicted probabilities and the one-hot
    actual outcome, summed across the 3 classes per match. Lower is better."""
    onehot = _onehot(outcomes)
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ranked_probability_score(probs: np.ndarray, outcomes) -> float:
    """Ranked Probability Score, respecting the ordinal structure of
    football outcomes (Away < Draw < Home) - unlike log loss/Brier, RPS
    penalizes a wrong prediction less if it was "close" in outcome-order
    (predicting a draw when the away team won is penalized less than
    predicting a home win when the away team won). Standard in football
    forecasting literature since Constantinou & Fenton (2012) argued for it
    over plain accuracy/log loss for this reason. Lower is better.

    Columns are ordered (H, D, A) throughout this module, but RPS requires
    genuine ordinal order - reordered internally to (A, D, H) before computing
    cumulative sums.
    """
    order = [2, 1, 0]  # (H, D, A) -> (A, D, H)
    ordered_probs = probs[:, order]
    onehot = _onehot(outcomes)[:, order]

    cum_probs = np.cumsum(ordered_probs, axis=1)
    cum_outcomes = np.cumsum(onehot, axis=1)
    # Final cumulative column is always 1.0 - 1.0 = 0 for both, so the
    # standard RPS normalization divides by (n_categories - 1).
    squared_diffs = (cum_probs - cum_outcomes) ** 2
    per_match_rps = np.sum(squared_diffs[:, :-1], axis=1) / (probs.shape[1] - 1)
    return float(np.mean(per_match_rps))


def evaluate_all(probs: np.ndarray, outcomes) -> dict:
    return {
        "log_loss": log_loss(probs, outcomes),
        "brier_score": brier_score(probs, outcomes),
        "rps": ranked_probability_score(probs, outcomes),
    }
