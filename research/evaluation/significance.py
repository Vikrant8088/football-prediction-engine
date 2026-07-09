"""Statistical significance testing between models' predictions.

A model beating another on mean log loss over a few thousand matches could
still be noise. This compares models on the *same* matches (paired) via two
tests on the per-match log-loss difference:

- Paired t-test (scipy.stats.ttest_rel): standard, assumes the differences
  are approximately normal.
- Wilcoxon signed-rank test (scipy.stats.wilcoxon): a non-parametric
  robustness check, since per-match log-loss differences are often skewed
  (a few very wrong confident predictions dominate the mean).

Both are reported rather than picking one, since they can disagree and
that disagreement is itself informative about how robust a claimed
difference is.
"""

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from research.evaluation.metrics import log_loss


def per_match_log_loss(predictions: pd.DataFrame) -> pd.DataFrame:
    """Wide DataFrame: one row per match, one column per model, values are
    that model's log-loss contribution for that specific match. Only
    matches every model actually predicted (an inner join) are included,
    so comparisons are always on an identical match set."""
    predictions = predictions.copy()
    predictions["match_key"] = (
        predictions["date"].astype(str) + "|" + predictions["home_team"] + "|" + predictions["away_team"]
    )

    eps = 1e-12
    picked_prob = np.select(
        [predictions["result"] == "H", predictions["result"] == "D", predictions["result"] == "A"],
        [predictions["p_home"], predictions["p_draw"], predictions["p_away"]],
    )
    predictions["match_log_loss"] = -np.log(np.clip(picked_prob, eps, 1.0))

    wide = predictions.pivot(index="match_key", columns="model", values="match_log_loss")
    return wide.dropna(axis=0, how="any")


def pairwise_significance(predictions: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """For every pair of models, paired-tests whether their per-match log
    loss differs significantly. Negative mean_diff means the first model
    in the pair has lower (better) log loss."""
    wide = per_match_log_loss(predictions)

    rows = []
    for model_a, model_b in combinations(models, 2):
        diff = wide[model_a] - wide[model_b]
        t_stat, t_pvalue = stats.ttest_rel(wide[model_a], wide[model_b])
        try:
            w_stat, w_pvalue = stats.wilcoxon(wide[model_a], wide[model_b])
        except ValueError:
            # wilcoxon raises if all differences are exactly zero
            w_stat, w_pvalue = float("nan"), float("nan")

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_matches": len(wide),
                "mean_log_loss_diff": float(diff.mean()),
                "paired_t_pvalue": float(t_pvalue),
                "wilcoxon_pvalue": float(w_pvalue),
            }
        )
    return pd.DataFrame(rows)
