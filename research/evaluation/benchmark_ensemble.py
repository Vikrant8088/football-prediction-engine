"""Phase 2 benchmark: does a weighted ensemble beat the best single model?

Phase 1 ended with two strong, structurally different models - Elo (rating
dynamics) and Dixon-Coles-xG (xG-based scoreline distribution) - neither
dominating the other. This benchmark adds an EnsembleModel that blends the
strong base models with walk-forward-fit weights, and asks the one question
that matters: does the blend beat Elo (the reigning champion, 0.9956 on this
window)?

Same discipline as every prior benchmark: identical Understat match set,
walk-forward expanding window, proper scoring rules + calibration +
significance. The ensemble fits its weights only on an inner temporal split of
each fold's training data, so it never sees the season it predicts.

Output is written to research/results/ with an `ensemble_` prefix.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data_warehouse.utils.logging_config import configure_logging
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR, run_walk_forward, summarize
from research.evaluation.calibration import plot_calibration_curve
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel
from research.experiments.ensemble import EnsembleModel
from research.experiments.poisson_xg import PoissonXGModel

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
MIN_TRAINING_SEASONS = 4

# The strong, diverse base models the ensemble blends. The dominated goal-based
# poisson / dixon_coles from Phase 1 are left out - their signal is already
# carried (better) by their xG twins.
ENSEMBLE_BASES = {
    "elo": EloModel,
    "poisson_xg": PoissonXGModel,
    "dixon_coles_xg": DixonColesXGModel,
}


def _build_ensemble():
    return EnsembleModel(ENSEMBLE_BASES)


MODEL_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "poisson_xg": PoissonXGModel,
    "dixon_coles_xg": DixonColesXGModel,
    "ensemble": _build_ensemble,
}


def _walk_forward_mean_weights(matches, min_training_seasons) -> dict:
    """Report the ensemble's average learned weights across the SAME expanding
    walk-forward folds the benchmark evaluates on - a faithful picture of how
    much it trusts each base model over time, not a single window (which can be
    unrepresentative: the most recent fold happens to collapse onto Elo alone)."""
    seasons = sorted(matches["season"].unique())
    per_fold = []
    for season in seasons[min_training_seasons:]:
        train = matches[matches["season"] < season]
        per_fold.append(_build_ensemble().fit(train).weights_)
    names = list(ENSEMBLE_BASES)
    return {n: float(np.mean([w[n] for w in per_fold])) for n in names}


def recommend(summary, significance, weights) -> str:
    ranked = summary.sort_values("log_loss")
    best_model = ranked.index[0]
    lines = [
        f"Best model by log loss: **{best_model}** "
        f"({ranked.loc[best_model, 'log_loss']:.4f}).",
        "Ensemble weights (averaged across walk-forward folds): "
        + ", ".join(f"{n} {w:.0%}" for n, w in weights.items())
        + ".",
    ]

    if "ensemble" in summary.index and "elo" in summary.index:
        ll_ens = summary.loc["ensemble", "log_loss"]
        ll_elo = summary.loc["elo", "log_loss"]
        row = significance[
            ((significance["model_a"] == "ensemble") & (significance["model_b"] == "elo"))
            | ((significance["model_a"] == "elo") & (significance["model_b"] == "ensemble"))
        ]
        sig_txt = ""
        if not row.empty:
            p_t = float(row.iloc[0]["paired_t_pvalue"])
            p_w = float(row.iloc[0]["wilcoxon_pvalue"])
            significant = p_t < 0.05 and p_w < 0.05
            sig_txt = (
                f" ({'significant' if significant else 'not significant'}: "
                f"paired t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"
            )
        if ll_ens < ll_elo:
            lines.append(
                f"**The ensemble beats Elo** ({ll_ens:.4f} vs {ll_elo:.4f}){sig_txt} "
                f"- the first model to dethrone Elo. Blending the rating and "
                f"scoreline views does better than either alone. New champion: "
                f"ensemble."
            )
        else:
            lines.append(
                f"The ensemble ({ll_ens:.4f}) does not beat Elo ({ll_elo:.4f})"
                f"{sig_txt}. Elo remains champion; the blend did not add over its "
                f"strongest component on this window."
            )
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    lines = [
        f"# Phase 2 Research Benchmark (ensemble) - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat). "
        f"Evaluated (walk-forward, expanding window) on "
        f"{run_config['evaluated_seasons'][0]} to {run_config['evaluated_seasons'][-1]} "
        f"({len(run_config['evaluated_seasons'])} seasons, "
        f"{int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "**Question:** does a walk-forward-weighted ensemble of the strong, "
        "diverse base models (Elo + Poisson-xG + Dixon-Coles-xG) beat the best "
        "single model, Elo? The ensemble fits its weights only on an inner "
        "temporal split of each fold's training data - it never sees the "
        "evaluated season.",
        "",
        "## Comparison",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "Lower is better for log loss, RPS, Brier score, and ECE.",
        "",
        "## Statistical significance (paired, per-match log loss)",
        "",
        significance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Next steps",
        "",
        "If the ensemble wins, it becomes the model to beat and the next gains "
        "come from adding genuinely new signals (Phase 3: injuries, lineups, "
        "rest) and an ML base model on a feature store. A remaining lever for "
        "any of the current models is hyperparameter tuning (Elo K / home "
        "advantage; Dixon-Coles xi), still on literature defaults.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_ensemble.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    logger.info("Loaded %d matches for Understat %s", len(matches), UNDERSTAT_LEAGUE)

    predictions, runtimes = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)
    summary = summarize(predictions, runtimes)
    significance = pairwise_significance(predictions, list(MODEL_BUILDERS.keys()))
    weights = _walk_forward_mean_weights(matches, MIN_TRAINING_SEASONS)
    recommendation = recommend(summary, significance, weights)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictions.to_csv(RESULTS_DIR / f"ensemble_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"ensemble_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"ensemble_significance_{run_id}.csv", index=False)

    all_seasons = sorted(matches["season"].unique().tolist())
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "source": "understat",
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "evaluated_seasons": all_seasons[MIN_TRAINING_SEASONS:],
        "models": list(MODEL_BUILDERS.keys()),
        "ensemble_bases": list(ENSEMBLE_BASES.keys()),
        "ensemble_mean_fold_weights": weights,
    }
    (RESULTS_DIR / f"ensemble_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in MODEL_BUILDERS:
        model_predictions = predictions[predictions["model"] == model_name]
        probs = model_predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = model_predictions["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            plot_path = RESULTS_DIR / f"ensemble_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, plot_path)
            plot_names.append(plot_path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"ensemble_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
