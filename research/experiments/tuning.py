"""Leakage-safe hyperparameter tuning by nested walk-forward search.

Every model in this project has so far used literature-default hyperparameters
(Elo K=20 / home advantage=100; Dixon-Coles xi=0.0065/day) - values from the
published papers, never fitted to this dataset. The Phase 0 report flagged this
as the outstanding milestone. This module closes it.

The one thing that makes hyperparameter tuning honest is WHERE the settings are
chosen. Picking them on the seasons you then report scores for is the classic
way to fool yourself: the reported number becomes the best of N tries, not an
out-of-sample estimate. So `TunedModel` never looks at the season it will
predict. On each `fit(train_df)` it:

    1. splits the TRAINING window by season into inner-train (older) +
       inner-validation (the most recent `inner_val_seasons`);
    2. fits every candidate setting on inner-train and scores it (log loss) on
       inner-validation;
    3. keeps the winner, and refits it on the full training window.

Nested inside the outer walk-forward, this means the settings used to predict
season S were chosen using only data from strictly before S - the same
discipline as the ensemble's weight fitting. `TunedModel` implements the
standard fit/predict_proba contract, so it drops into the harness (and into
`EnsembleModel` as a base) like any other model, and it reports the settings it
chose (`best_params_`), keeping it explainable.
"""

import itertools
import logging

import numpy as np
import pandas as pd

from research.evaluation.metrics import log_loss
from research.experiments.base import PredictionModel
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel

logger = logging.getLogger(__name__)

# Search grids, centred on the literature defaults so tuning can only improve
# on (or reproduce) the published choice.
ELO_GRID = {
    "k_factor": [10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
    "home_advantage": [40.0, 60.0, 80.0, 100.0, 120.0],
}
ELO_DEFAULTS = {"k_factor": 20.0, "home_advantage": 100.0}

# xi=0 means "no time decay" (all history weighted equally).
DIXON_COLES_XG_GRID = {"xi": [0.0, 0.002, 0.0065, 0.012]}
DIXON_COLES_XG_DEFAULTS = {"xi": 0.0065}


class TunedModel(PredictionModel):
    """Wraps a model class, choosing its hyperparameters by an inner temporal
    split of the training data only."""

    def __init__(self, build, param_grid, defaults, inner_val_seasons=2, label=""):
        """`build(**params)` returns an unfitted PredictionModel."""
        self.build = build
        self.param_grid = dict(param_grid)
        self.defaults = dict(defaults)
        self.inner_val_seasons = inner_val_seasons
        self.label = label
        self.best_params_ = None
        self._model = None

    def _candidates(self):
        names = list(self.param_grid)
        for values in itertools.product(*(self.param_grid[n] for n in names)):
            yield dict(zip(names, values))

    def _select_params(self, train_df: pd.DataFrame) -> dict:
        seasons = sorted(train_df["season"].unique())
        if len(seasons) <= self.inner_val_seasons:
            # Not enough history to hold seasons out - fall back to the
            # published defaults rather than tuning on data we also train on.
            return dict(self.defaults)

        val_seasons = set(seasons[-self.inner_val_seasons :])
        inner_train = train_df[~train_df["season"].isin(val_seasons)]
        inner_val = train_df[train_df["season"].isin(val_seasons)]
        fixtures = inner_val[["home_team", "away_team"]]
        outcomes = inner_val["result"].to_numpy()

        best_params, best_loss = None, np.inf
        for params in self._candidates():
            probs = self.build(**params).fit(inner_train).predict_proba(fixtures)
            loss = log_loss(probs, outcomes)
            if loss < best_loss:
                best_params, best_loss = params, loss
        return best_params

    def fit(self, train_df: pd.DataFrame) -> "TunedModel":
        self.best_params_ = self._select_params(train_df)
        logger.info(
            "Tuned %s -> %s",
            self.label or self.build.__name__,
            ", ".join(f"{k}={v}" for k, v in self.best_params_.items()),
        )
        self._model = self.build(**self.best_params_).fit(train_df)
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict_proba()")
        return self._model.predict_proba(fixtures_df)


def build_tuned_elo() -> TunedModel:
    return TunedModel(EloModel, ELO_GRID, ELO_DEFAULTS, label="elo")


def build_tuned_dixon_coles_xg() -> TunedModel:
    return TunedModel(
        DixonColesXGModel,
        DIXON_COLES_XG_GRID,
        DIXON_COLES_XG_DEFAULTS,
        label="dixon_coles_xg",
    )
