"""Shared interface every benchmark model implements.

The benchmark harness (evaluation/benchmark.py) iterates models through this
one contract - fit on a training window, then predict probabilities for a
held-out window - so adding a fifth model later never requires touching the
harness (the same lesson learned in the ingest layer's source abstraction).
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


class PredictionModel(ABC):
    """A model that predicts P(home win), P(draw), P(away win) for matches.

    `fit` is always called with the complete history available up to (but
    excluding) the evaluation window - for walk-forward evaluation, the
    harness calls `fit` once per fold with a progressively larger training
    set, never the other way round.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "PredictionModel":
        """`train_df` has columns: date, home_team, away_team, home_goals,
        away_goals, result (H/D/A), sorted chronologically. Returns self."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        """`fixtures_df` has columns: home_team, away_team (results are not
        passed in - a model must not be able to peek at what it's predicting).
        Returns an (n, 3) array of probabilities, columns ordered (H, D, A),
        each row summing to 1.0.
        """
        raise NotImplementedError
