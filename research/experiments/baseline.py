"""The null model: ignores both teams entirely and predicts the training
set's overall H/D/A frequency for every match.

This exists purely as the floor every other model must clear. A model that
can't beat this is not adding predictive value, whatever else it does.
"""

import numpy as np
import pandas as pd

from research.experiments.base import OUTCOMES, PredictionModel


class BaselineFrequencyModel(PredictionModel):
    def __init__(self):
        self._probabilities = None  # (p_H, p_D, p_A)

    def fit(self, train_df: pd.DataFrame) -> "BaselineFrequencyModel":
        counts = train_df["result"].value_counts()
        total = len(train_df)
        self._probabilities = np.array(
            [counts.get(outcome, 0) / total for outcome in OUTCOMES]
        )
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._probabilities is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        return np.tile(self._probabilities, (len(fixtures_df), 1))
