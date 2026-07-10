"""The production prediction engine: fit the champion, predict a real fixture.

Everything in `research/` exists to answer "which model is best, and how good is
it really?". This module is the other side of that: it takes the model that won,
trains it on all available history, and turns one fixture into a complete,
explainable forecast.

What comes out is deliberately more than a scoreline:

- the champion's Win/Draw/Loss probabilities;
- a full scoreline distribution, whose marginals ARE those probabilities;
- every derived market (exact score, over/under, both-teams-to-score, ...),
  all consistent with each other because they share one grid;
- a confidence tier annotated with the accuracy that confidence has HISTORICALLY
  delivered, not one that is promised;
- an explanation: how much the blend trusted each base model, and what each of
  them individually thought.

The last two are the point. An engine that says "62%, and when I say 62% I have
been right 65% of the time, and here is why I think it" is worth more than one
that says "2-1".

Models are imported from `research.experiments` rather than copied here: the
champion is promoted by reference so production can never silently diverge from
the thing that was actually benchmarked.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prediction_engine import confidence as confidence_module
from prediction_engine import markets as markets_module
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from research.data.xg_loader import load_understat_matches

logger = logging.getLogger(__name__)

OUTCOME_LABELS = {"H": "home win", "D": "draw", "A": "away win"}


@dataclass
class Prediction:
    """One fixture, fully forecast."""

    league: str
    home_team: str
    away_team: str

    probabilities: Dict[str, float]          # {"home": .., "draw": .., "away": ..}
    top_pick: str                            # "H" | "D" | "A"
    confidence: Dict[str, object]            # tier + backtested accuracy
    publishable: bool                        # clears the confidence threshold?

    scoreline_grid: np.ndarray = field(repr=False)
    markets: Dict[str, object] = field(repr=False)

    model_weights: Dict[str, float] = field(default_factory=dict)
    base_views: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)

    @property
    def most_likely_score(self) -> Tuple[Tuple[int, int], float]:
        return self.markets["most_likely_scorelines"][0]

    def summary(self) -> str:
        (home_goals, away_goals), score_p = self.most_likely_score
        conf = self.confidence
        lines = [
            f"{self.home_team} vs {self.away_team}  ({self.league})",
            "",
            f"  Home win  {self.probabilities['home']:6.1%}",
            f"  Draw      {self.probabilities['draw']:6.1%}",
            f"  Away win  {self.probabilities['away']:6.1%}",
            "",
            f"  Most likely score : {home_goals}-{away_goals}  ({score_p:.1%})",
            f"  Expected goals    : {self.markets['expected_goals']['home']:.2f} - "
            f"{self.markets['expected_goals']['away']:.2f}",
            f"  Over 2.5 goals    : {self.markets['over_under_2_5']['over']:.1%}",
            f"  Both teams score  : {self.markets['both_teams_to_score']['yes']:.1%}",
            "",
            f"  Call: {OUTCOME_LABELS[self.top_pick]} @ {conf['confidence']:.1%} "
            f"({conf['tier']} confidence)",
            f"  Historically, calls at this confidence were right "
            f"{conf['backtested_accuracy']:.1%} of the time.",
            f"  {'PUBLISH' if self.publishable else 'HOLD - below the confidence threshold'}",
            "",
            "  Why: blend weights " + ", ".join(
                f"{n} {w:.0%}" for n, w in sorted(
                    self.model_weights.items(), key=lambda kv: -kv[1]
                )
            ),
        ]
        for name, view in self.base_views.items():
            lines.append(
                f"    - {name:<16} H {view[0]:.0%}  D {view[1]:.0%}  A {view[2]:.0%}"
            )
        return "\n".join(lines)


class PredictionEngine:
    """Trains the champion once, then predicts any number of fixtures."""

    def __init__(self, league: str, model: ScorelineEnsemble, matches: pd.DataFrame):
        self.league = league
        self._model = model
        self._matches = matches

    @classmethod
    def train(cls, league: str = "EPL") -> "PredictionEngine":
        """Fit the champion on every match available for `league`."""
        matches = load_understat_matches(league)
        logger.info("Training champion on %d %s matches", len(matches), league)
        model = ScorelineEnsemble().fit(matches)
        return cls(league, model, matches)

    @property
    def teams(self) -> List[str]:
        return sorted(set(self._matches["home_team"]) | set(self._matches["away_team"]))

    @property
    def matches(self) -> pd.DataFrame:
        """The history the champion was trained on."""
        return self._matches

    def scoreline_grid(self, home_team: str, away_team: str) -> np.ndarray:
        """P(home_goals=h, away_goals=a) for a fixture. Its home/draw/away
        regions sum to the champion's 1X2 probabilities exactly."""
        self._check_team(home_team)
        self._check_team(away_team)
        return self._model.scoreline_grid(home_team, away_team)

    def _check_team(self, team: str) -> None:
        if team not in set(self.teams):
            raise ValueError(
                f"'{team}' is not a team in {self.league}. Closest known: "
                + ", ".join(t for t in self.teams if team.lower() in t.lower()) or "none"
            )

    def predict(
        self, home_team: str, away_team: str, threshold: Optional[float] = None
    ) -> Prediction:
        self._check_team(home_team)
        self._check_team(away_team)
        if home_team == away_team:
            raise ValueError("A team cannot play itself")

        threshold = confidence_module.DEFAULT_THRESHOLD if threshold is None else threshold

        grid = self._model.scoreline_grid(home_team, away_team)
        all_markets = markets_module.all_markets(grid)
        outcome = all_markets["outcome"]

        ordered = [("H", outcome["home"]), ("D", outcome["draw"]), ("A", outcome["away"])]
        top_pick, top_probability = max(ordered, key=lambda kv: kv[1])

        return Prediction(
            league=self.league,
            home_team=home_team,
            away_team=away_team,
            probabilities=outcome,
            top_pick=top_pick,
            confidence=confidence_module.classify(top_probability),
            publishable=confidence_module.is_confident(top_probability, threshold),
            scoreline_grid=grid,
            markets=all_markets,
            model_weights=dict(self._model.weights_),
            base_views={
                name: tuple(float(x) for x in view)
                for name, view in self._model.base_views(home_team, away_team).items()
            },
        )
