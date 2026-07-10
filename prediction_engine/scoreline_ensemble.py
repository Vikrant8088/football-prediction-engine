"""The champion, extended to emit a full scoreline distribution.

The ensemble that won every benchmark blends three base models into a single
Win/Draw/Loss vector. But Elo - which carries ~72% of that blend's weight -
produces only a 3-way probability; it has no notion of a scoreline. So the
champion, as a 1X2 forecaster, cannot by itself say "2-1".

This resolves that cleanly rather than throwing away either half:

1. The **shape** of the scoreline distribution comes from the two models that
   genuinely model goals - Poisson-xG and Dixon-Coles-xG - blended using their
   own ensemble weights (renormalised between them).
2. The **outcome probabilities** come from the full champion ensemble, which is
   the better-calibrated 1X2 forecaster (and better calibrated than Pinnacle on
   home wins and draws).
3. The grid is then **rescaled region by region** - home-win cells, draw cells,
   away-win cells - so that each region sums to exactly the champion's P(H),
   P(D), P(A).

The result satisfies the invariant that matters: *the scoreline grid's
marginals are the champion's 1X2 probabilities, exactly.* Every derived market
(exact score, over/under, both-teams-to-score) is therefore consistent with the
headline forecast, and inherits its calibration. This is the project's founding
principle finally made whole - "exact score prediction is a consequence of
probability modelling".

Edge case handled: the ensemble occasionally puts 100% of its weight on Elo
(it did exactly this in the most recent Premier League fold). The grid weights
would then be 0/0, so we fall back to weighting the two scoreline models
equally - the outcome probabilities are still the champion's, so only the
shape within each region is affected.
"""

import numpy as np
import pandas as pd

from research.experiments.base import PredictionModel
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel
from research.experiments.ensemble import EnsembleModel
from research.experiments.poisson_xg import PoissonXGModel

# Base models that can produce a scoreline grid. Elo cannot.
GRID_MODELS = ("poisson_xg", "dixon_coles_xg")

BASE_BUILDERS = {
    "elo": EloModel,
    "poisson_xg": PoissonXGModel,
    "dixon_coles_xg": DixonColesXGModel,
}


def outcome_masks(size: int):
    """Boolean masks selecting the home-win / draw / away-win cells of a
    (size x size) scoreline grid indexed [home_goals, away_goals]."""
    goals = np.arange(size)
    home = goals[:, None] > goals[None, :]
    draw = goals[:, None] == goals[None, :]
    away = goals[:, None] < goals[None, :]
    return home, draw, away


class ScorelineEnsemble(PredictionModel):
    """The champion ensemble, plus a consistent scoreline grid."""

    def __init__(self):
        self._ensemble = EnsembleModel(BASE_BUILDERS)

    @property
    def weights_(self) -> dict:
        return self._ensemble.weights_

    @property
    def base_models(self) -> dict:
        return self._ensemble.base_models

    def fit(self, train_df: pd.DataFrame) -> "ScorelineEnsemble":
        self._ensemble.fit(train_df)
        return self

    def predict_proba(self, fixtures_df: pd.DataFrame) -> np.ndarray:
        return self._ensemble.predict_proba(fixtures_df)

    def _grid_weights(self) -> dict:
        weights = {n: self.weights_[n] for n in GRID_MODELS if n in self.weights_}
        total = sum(weights.values())
        if total <= 1e-9:
            # The blend gave the scoreline models no weight at all (all on Elo).
            # Fall back to an equal split: the outcome probabilities are still
            # the champion's, so only the within-region shape is affected.
            return {n: 1.0 / len(GRID_MODELS) for n in GRID_MODELS}
        return {n: w / total for n, w in weights.items()}

    def scoreline_grid(self, home_team: str, away_team: str) -> np.ndarray:
        """P(home_goals=h, away_goals=a), rescaled so the grid's home/draw/away
        regions sum to the champion ensemble's P(H), P(D), P(A) exactly."""
        bases = self.base_models
        grid = sum(
            w * bases[name].score_grid(home_team, away_team)
            for name, w in self._grid_weights().items()
        )
        grid = np.clip(np.asarray(grid, dtype=float), 0.0, None)
        grid /= grid.sum()

        probabilities = self.predict_proba(
            pd.DataFrame({"home_team": [home_team], "away_team": [away_team]})
        )[0]

        rescaled = np.zeros_like(grid)
        for mask, target in zip(outcome_masks(grid.shape[0]), probabilities):
            mass = grid[mask].sum()
            if mass > 1e-12:
                rescaled[mask] = grid[mask] * (target / mass)
            else:
                # Degenerate: this region has no mass at all. Spread the
                # champion's probability for it evenly rather than losing it.
                rescaled[mask] = target / mask.sum()
        return rescaled / rescaled.sum()

    def base_views(self, home_team: str, away_team: str) -> dict:
        """Each base model's own 1X2 opinion - the explanation behind the blend."""
        fixture = pd.DataFrame({"home_team": [home_team], "away_team": [away_team]})
        return {
            name: model.predict_proba(fixture)[0]
            for name, model in self.base_models.items()
        }
