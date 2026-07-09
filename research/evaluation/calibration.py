"""Calibration: are predicted probabilities honest?

If a model says "60% home win" across many matches, roughly 60% of those
matches should actually end in a home win. Log loss/Brier/RPS score
sharpness and accuracy together; this module checks calibration on its own,
since a model can have a good average score while still being
systematically over- or under-confident.
"""

from typing import Optional

import numpy as np
import pandas as pd

from research.evaluation.metrics import OUTCOME_ORDER


def calibration_table(probs: np.ndarray, outcomes, outcome_label: str, n_bins: int = 10) -> pd.DataFrame:
    """Bins predicted P(outcome_label) into `n_bins` equal-width buckets and
    compares the mean predicted probability against the observed frequency
    of that outcome within each bucket.

    Returns a DataFrame with columns: bin_low, bin_high, n_matches,
    mean_predicted, observed_frequency. Bins with zero matches are omitted.
    """
    outcome_idx = OUTCOME_ORDER.index(outcome_label)
    predicted = probs[:, outcome_idx]
    actual = np.array([1.0 if o == outcome_label else 0.0 for o in outcomes])

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(predicted, bin_edges[1:-1]), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_matches = int(mask.sum())
        if n_matches == 0:
            continue
        rows.append(
            {
                "bin_low": bin_edges[b],
                "bin_high": bin_edges[b + 1],
                "n_matches": n_matches,
                "mean_predicted": float(predicted[mask].mean()),
                "observed_frequency": float(actual[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(probs: np.ndarray, outcomes, outcome_label: str, n_bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed frequency
    across bins - a single number summarizing calibration_table. Lower is
    better; 0 is perfect calibration."""
    table = calibration_table(probs, outcomes, outcome_label, n_bins)
    if table.empty:
        return float("nan")
    weights = table["n_matches"] / table["n_matches"].sum()
    gaps = (table["mean_predicted"] - table["observed_frequency"]).abs()
    return float((weights * gaps).sum())


def plot_calibration_curve(probs: np.ndarray, outcomes, outcome_label: str, output_path, n_bins: int = 10) -> None:
    """Saves a reliability diagram (predicted vs. observed frequency, plus
    the perfect-calibration diagonal) to `output_path`. Requires matplotlib;
    only imported here so the rest of this module has no hard dependency on
    a plotting backend being available."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = calibration_table(probs, outcomes, outcome_label, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.scatter(table["mean_predicted"], table["observed_frequency"], s=table["n_matches"])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration: P({outcome_label})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
