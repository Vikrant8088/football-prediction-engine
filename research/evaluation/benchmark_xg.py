"""Phase 1 benchmark: does fitting team strength on Expected Goals (xG) beat
fitting it on actual goals?

This is the reproducible experiment behind Feature Roadmap Phase 1. It runs the
exact same walk-forward, expanding-window discipline as the Phase 0 benchmark
(research/evaluation/benchmark.py), but on the Understat xG dataset and with
one extra model - PoissonXG - added to the goal-based field.

Fair-comparison design: every model is fit and evaluated on the *same*
Understat matches (Understat carries both goals and xG, so no cross-source
join is needed). The goal-based models (baseline/elo/poisson/dixon_coles) use
goals; PoissonXG uses xG. Any difference in out-of-sample score is therefore
attributable to the goals-vs-xG signal, not to a different match set or data
provider. Because Understat's history starts at 2014/15, this is a *different*
(shorter) evaluation window than Phase 0, so the goal models are re-run here to
establish the champion-to-beat on exactly this window - the Phase 0 log-loss
numbers are NOT directly comparable and are not reused.

Running this script is the whole experiment. Output (predictions, summary,
significance, config, calibration plots, and a human-readable report) is
written to research/results/ with an `xg_` prefix, never just printed.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from data_warehouse.utils.logging_config import configure_logging
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR, run_walk_forward, summarize
from research.evaluation.calibration import plot_calibration_curve
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.dixon_coles import DixonColesModel
from research.experiments.elo import EloModel
from research.experiments.poisson import PoissonModel
from research.experiments.poisson_xg import PoissonXGModel

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
# Understat has 12 seasons (2014/15-2025/26). Train-only on the first 4, then
# evaluate walk-forward on the remaining 8 - the same ~45% train-only split
# ratio the Phase 0 benchmark used (8 of 17 seasons).
MIN_TRAINING_SEASONS = 4

MODEL_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "poisson": PoissonModel,
    "dixon_coles": DixonColesModel,
    "poisson_xg": PoissonXGModel,
}


def recommend(summary, significance) -> str:
    """Phase 1's headline question is narrower than Phase 0's: did the xG model
    (poisson_xg) beat its direct goal-based twin (poisson), and did it beat the
    overall field? Both are reported honestly, significance included."""
    ranked = summary.sort_values("log_loss")
    best_model = ranked.index[0]

    lines = [
        f"Best model by log loss: **{best_model}** "
        f"({ranked.loc[best_model, 'log_loss']:.4f})."
    ]

    # The clean, controlled contrast: same model family, goals vs xG.
    if "poisson" in summary.index and "poisson_xg" in summary.index:
        ll_goals = summary.loc["poisson", "log_loss"]
        ll_xg = summary.loc["poisson_xg", "log_loss"]
        better = "xG" if ll_xg < ll_goals else "goals"
        row = significance[
            (
                (significance["model_a"] == "poisson")
                & (significance["model_b"] == "poisson_xg")
            )
            | (
                (significance["model_a"] == "poisson_xg")
                & (significance["model_b"] == "poisson")
            )
        ]
        sig_txt = ""
        if not row.empty:
            p_t = float(row.iloc[0]["paired_t_pvalue"])
            p_w = float(row.iloc[0]["wilcoxon_pvalue"])
            significant = p_t < 0.05 and p_w < 0.05
            sig_txt = (
                f" The gap is {'significant' if significant else 'NOT significant'} "
                f"(paired t p={p_t:.4f}, Wilcoxon p={p_w:.4f})."
            )
        lines.append(
            f"Controlled goals-vs-xG contrast (same Poisson machinery): "
            f"poisson (goals) log loss {ll_goals:.4f} vs poisson_xg log loss "
            f"{ll_xg:.4f} - fitting strength on **{better}** predicts real "
            f"results better on this window.{sig_txt}"
        )

    if best_model == "poisson_xg":
        lines.append(
            "The xG model is the outright best model on this window - Phase 1 "
            "hypothesis supported; promote poisson_xg to the candidate pool."
        )
    else:
        lines.append(
            f"The xG model is not the outright best here ({best_model} leads). "
            f"Record the result honestly and treat xG as one useful signal to "
            f"combine in Phase 2 rather than a standalone winner."
        )
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    lines = [
        f"# Phase 1 Research Benchmark (xG) - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat). "
        f"Trained on seasons {run_config['seasons_available'][0]} onward; "
        f"evaluated (walk-forward, expanding window) on "
        f"{run_config['evaluated_seasons'][0]} to {run_config['evaluated_seasons'][-1]} "
        f"({len(run_config['evaluated_seasons'])} seasons, "
        f"{int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "**Question:** does estimating team attack/defense strength from "
        "Expected Goals (xG) predict real match outcomes better than estimating "
        "it from actual goals? `poisson_xg` is the xG twin of `poisson`; every "
        "model is scored on the identical Understat match set, so the "
        "goals-vs-xG contrast is controlled.",
        "",
        "This is a different, shorter window than the Phase 0 benchmark "
        "(Understat xG starts at 2014/15), so the goal models are re-run here "
        "to set the champion-to-beat on this window; Phase 0 numbers are not "
        "directly comparable.",
        "",
        "## Comparison",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "Lower is better for log loss, RPS, Brier score, and ECE (expected "
        "calibration error). Runtime is wall-clock seconds summed across all "
        "walk-forward folds on this machine.",
        "",
        "## Statistical significance (paired, per-match log loss)",
        "",
        significance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "p < 0.05 on both the paired t-test and Wilcoxon signed-rank test is "
        "treated as significant.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Next steps",
        "",
        "Two threads follow from this result, both under the same walk-forward "
        "discipline: (1) an xG-aware Dixon-Coles (xG strengths + the low-score "
        "correlation correction and time decay), and (2) Phase 2 - combining "
        "the goal and xG signals in a feature store + ensemble rather than "
        "choosing one. Hyperparameters remain literature defaults (Elo K/home "
        "advantage; Dixon-Coles xi) and are still a pending tuning pass.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_xg.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    logger.info(
        "Loaded %d matches across %d seasons for Understat %s",
        len(matches),
        matches["season"].nunique(),
        UNDERSTAT_LEAGUE,
    )

    predictions, runtimes = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)
    summary = summarize(predictions, runtimes)
    significance = pairwise_significance(predictions, list(MODEL_BUILDERS.keys()))
    recommendation = recommend(summary, significance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    predictions.to_csv(RESULTS_DIR / f"xg_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"xg_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"xg_significance_{run_id}.csv", index=False)

    all_seasons = sorted(matches["season"].unique().tolist())
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "source": "understat",
        "seasons_available": all_seasons,
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "evaluated_seasons": all_seasons[MIN_TRAINING_SEASONS:],
        "models": list(MODEL_BUILDERS.keys()),
    }
    (RESULTS_DIR / f"xg_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in MODEL_BUILDERS:
        model_predictions = predictions[predictions["model"] == model_name]
        probs = model_predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = model_predictions["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            plot_path = RESULTS_DIR / f"xg_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, plot_path)
            plot_names.append(plot_path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"xg_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
