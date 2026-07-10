"""Selective prediction: only speak when the engine is actually confident.

Top-pick accuracy over ALL matches is capped near 55% by football itself - a
quarter of matches are draws, and a draw is essentially never the single most
likely outcome (it was the top pick in 0 of 2,660 Pinnacle-priced matches). So
"we're right 51% of the time" understates a well-calibrated engine badly.

The honest way to a higher hit rate is not a better model - it is *saying less*.
Because this engine's probabilities are truthful (verified: better calibrated
than Pinnacle on home wins and draws), its own confidence is a reliable filter.
Publish only the calls above a threshold and the hit rate rises exactly as the
calibration predicts, at the cost of covering fewer matches.

The accuracies below are MEASURED, not promised - from the 14,284-match,
five-league walk-forward run `multileague_predictions_20260710T052635Z`, where a
prediction's "confidence" is the probability of its top pick:

    confidence     matches   share    accuracy
    >= 70%           1,565    11.0%     72.6%
    60-70%           2,064    14.4%     65.1%
    50-60%           3,468    24.3%     53.1%
    40-50%           5,079    35.6%     43.5%
    <  40%           2,108    14.8%     36.9%

Read the rows as: "when the engine says 60-70%, it is right about 65% of the
time." They line up - that is the calibration, restated as a promise you can
hold it to.

These constants MUST be recomputed whenever the champion changes; `recompute_tiers`
does that from a predictions CSV so they can never silently drift out of date.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

MEASUREMENT_RUN = "multileague_predictions_20260710T052635Z"

# (lower_bound, upper_bound, label, measured accuracy, measured share of matches)
TIERS: List[Tuple[float, float, str, float, float]] = [
    (0.00, 0.40, "very low", 0.369, 0.148),
    (0.40, 0.50, "low", 0.435, 0.356),
    (0.50, 0.60, "medium", 0.531, 0.243),
    (0.60, 0.70, "high", 0.651, 0.144),
    (0.70, 1.01, "very high", 0.726, 0.110),
]

# Above this, the engine is right ~2 times in 3. A sensible default for
# "publish a confident call"; the coverage cost is stated in `coverage_at`.
DEFAULT_THRESHOLD = 0.60


def classify(top_probability: float) -> Dict[str, object]:
    """Label a prediction's confidence and attach the accuracy that level of
    confidence has historically delivered."""
    for low, high, label, accuracy, share in TIERS:
        if low <= top_probability < high:
            return {
                "confidence": float(top_probability),
                "tier": label,
                "backtested_accuracy": accuracy,
                "share_of_matches": share,
                "measurement_run": MEASUREMENT_RUN,
            }
    raise ValueError(f"top_probability {top_probability} outside [0, 1]")


def is_confident(top_probability: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Should the engine publish a call at all?"""
    return top_probability >= threshold


def coverage_at(threshold: float) -> float:
    """Share of matches the engine would call at a given threshold - the honest
    cost of a higher hit rate."""
    return float(sum(share for low, _, _, _, share in TIERS if low >= threshold))


def recompute_tiers(predictions: pd.DataFrame, model: str = "ensemble") -> pd.DataFrame:
    """Recompute the tier table from a walk-forward predictions frame.

    Guards against the constants above silently going stale after a model
    change. `predictions` needs columns: model, result, p_home, p_draw, p_away.
    """
    df = predictions[predictions["model"] == model]
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
    picks = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
    correct = picks == df["result"].to_numpy()
    confidence = probs.max(axis=1)

    rows = []
    for low, high, label, _, _ in TIERS:
        in_tier = (confidence >= low) & (confidence < high)
        if not in_tier.any():
            continue
        rows.append({
            "tier": label,
            "low": low,
            "high": high,
            "n_matches": int(in_tier.sum()),
            "share_of_matches": float(in_tier.mean()),
            "accuracy": float(correct[in_tier].mean()),
        })
    return pd.DataFrame(rows)
