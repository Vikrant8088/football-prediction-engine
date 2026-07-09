"""Walk-forward benchmark: the reproducible experiment entry point for
Research Milestone 1.

Evaluation design (expanding window, by season): for each season from
MIN_TRAINING_SEASONS onward, every model is fit on all seasons strictly
before it and evaluated on that season's matches, then the window expands
to include that season and rolls forward. This is standard practice in the
forecasting literature (Dixon & Coles themselves evaluate this way) and,
critically, never lets a model see a result before predicting it - the same
no-shuffling-across-time discipline the raw ingest layer's versioning was
built to support in the first place.

Running this script is the whole experiment: same code, same data (whatever
the versioned raw lake currently holds for the configured league/seasons),
same result. Output is written to research/results/ - both a machine-
readable CSV/JSON and a human-readable comparison report - never just
printed and discarded.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.loader import load_league_matches
from research.evaluation.calibration import expected_calibration_error, plot_calibration_curve
from research.evaluation.metrics import evaluate_all
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.dixon_coles import DixonColesModel
from research.experiments.elo import EloModel
from research.experiments.poisson import PoissonModel

logger = logging.getLogger(__name__)

LEAGUE_CODE = "E0"
MIN_TRAINING_SEASONS = 8  # first 8 seasons are never evaluated on, only trained on
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MODEL_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "poisson": PoissonModel,
    "dixon_coles": DixonColesModel,
}


def run_walk_forward(
    matches: pd.DataFrame,
    model_builders: dict = MODEL_BUILDERS,
    min_training_seasons: int = MIN_TRAINING_SEASONS,
):
    """Returns (predictions, runtimes).

    predictions: long-format DataFrame, one row per (model, season, match),
    with columns date, season, home_team, away_team, result, model,
    p_home, p_draw, p_away - the raw material every metric/report is
    computed from, kept so any metric can be recomputed later without
    re-running the (slow) model fits.

    runtimes: {model_name: {"fit_seconds": total, "predict_seconds": total,
    "n_folds": count}} - wall-clock cost, measured (not estimated), summed
    across every walk-forward fold.

    `model_builders` / `min_training_seasons` are parameters (defaulting to
    this module's Phase 0 constants) so alternative experiments - e.g. the
    Phase 1 xG benchmark on a different dataset and model set - reuse the exact
    same walk-forward discipline rather than reimplementing it.
    """
    seasons = sorted(matches["season"].unique())
    eval_seasons = seasons[min_training_seasons:]

    all_predictions = []
    runtimes = defaultdict(lambda: {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 0})

    for season in eval_seasons:
        train_df = matches[matches["season"] < season]
        test_df = matches[matches["season"] == season]
        fixtures = test_df[["home_team", "away_team"]]

        for model_name, build_model in model_builders.items():
            logger.info("Fitting %s on %d matches, evaluating on %s (%d matches)",
                        model_name, len(train_df), season, len(test_df))
            model = build_model()

            fit_start = time.perf_counter()
            model.fit(train_df)
            fit_seconds = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            probs = model.predict_proba(fixtures)
            predict_seconds = time.perf_counter() - predict_start

            runtimes[model_name]["fit_seconds"] += fit_seconds
            runtimes[model_name]["predict_seconds"] += predict_seconds
            runtimes[model_name]["n_folds"] += 1

            fold_result = test_df[["date", "season", "home_team", "away_team", "result"]].copy()
            fold_result["model"] = model_name
            fold_result["p_home"] = probs[:, 0]
            fold_result["p_draw"] = probs[:, 1]
            fold_result["p_away"] = probs[:, 2]
            all_predictions.append(fold_result)

    return pd.concat(all_predictions, ignore_index=True), dict(runtimes)


def summarize(predictions: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """One row per model: metrics computed over every out-of-sample
    prediction across the whole walk-forward run (not per-season averages
    of per-season metrics, which would weight small/large seasons unevenly),
    plus measured total and per-fold runtime."""
    rows = []
    for model_name, group in predictions.groupby("model"):
        probs = group[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = group["result"].to_numpy()
        metrics = evaluate_all(probs, outcomes)
        metrics["ece_home"] = expected_calibration_error(probs, outcomes, "H")
        metrics["ece_draw"] = expected_calibration_error(probs, outcomes, "D")
        metrics["ece_away"] = expected_calibration_error(probs, outcomes, "A")
        metrics["n_predictions"] = len(group)

        model_runtime = runtimes[model_name]
        total_seconds = model_runtime["fit_seconds"] + model_runtime["predict_seconds"]
        metrics["total_runtime_seconds"] = total_seconds
        metrics["mean_fit_seconds_per_fold"] = model_runtime["fit_seconds"] / model_runtime["n_folds"]

        metrics["model"] = model_name
        rows.append(metrics)

    summary = pd.DataFrame(rows).set_index("model")
    return summary[
        [
            "n_predictions",
            "log_loss",
            "rps",
            "brier_score",
            "ece_home",
            "ece_draw",
            "ece_away",
            "total_runtime_seconds",
            "mean_fit_seconds_per_fold",
        ]
    ].sort_values("log_loss")


def recommend_version_1(summary: pd.DataFrame, significance: pd.DataFrame) -> str:
    """Primary criterion is log loss (the metric that most directly measures
    calibrated probabilistic accuracy, per the earlier ML-framing decision to
    treat proper scoring rules, not accuracy, as the headline number).
    RPS/Brier agreement and calibration are cross-checked; statistical
    significance against the runner-up is reported honestly, including when
    it does NOT reach significance."""
    ranked = summary.sort_values("log_loss")
    best_model = ranked.index[0]
    runner_up = ranked.index[1] if len(ranked) > 1 else None

    lines = [f"Best model by log loss: **{best_model}** ({ranked.loc[best_model, 'log_loss']:.4f})."]

    rps_ranked = summary.sort_values("rps").index[0]
    brier_ranked = summary.sort_values("brier_score").index[0]
    if rps_ranked == best_model and brier_ranked == best_model:
        lines.append(f"{best_model} is also best on RPS and Brier score - all three proper scoring rules agree.")
    else:
        lines.append(
            f"Note: RPS ranks {rps_ranked} best and Brier ranks {brier_ranked} best - "
            f"metrics disagree, so this is not a clean sweep."
        )

    if runner_up is not None:
        row = significance[
            ((significance["model_a"] == best_model) & (significance["model_b"] == runner_up))
            | ((significance["model_a"] == runner_up) & (significance["model_b"] == best_model))
        ]
        if not row.empty:
            p_t = row.iloc[0]["paired_t_pvalue"]
            p_w = row.iloc[0]["wilcoxon_pvalue"]
            significant = p_t < 0.05 and p_w < 0.05
            lines.append(
                f"Margin over runner-up ({runner_up}) is "
                f"{'statistically significant' if significant else 'NOT statistically significant'} "
                f"at alpha=0.05 (paired t-test p={p_t:.4f}, Wilcoxon p={p_w:.4f})."
            )
            if not significant:
                lines.append(
                    f"Given the margin over {runner_up} is not significant, treat {best_model} as the "
                    f"provisional Version 1 candidate rather than a settled choice - more data or the "
                    f"planned hyperparameter tuning pass may change this ranking."
                )

    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_league_matches(LEAGUE_CODE)
    logger.info("Loaded %d matches across %d seasons for %s", len(matches), matches["season"].nunique(), LEAGUE_CODE)

    predictions, runtimes = run_walk_forward(matches)
    summary = summarize(predictions, runtimes)
    significance = pairwise_significance(predictions, list(MODEL_BUILDERS.keys()))
    recommendation = recommend_version_1(summary, significance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    predictions_path = RESULTS_DIR / f"predictions_{run_id}.csv"
    summary_path = RESULTS_DIR / f"summary_{run_id}.csv"
    significance_path = RESULTS_DIR / f"significance_{run_id}.csv"
    config_path = RESULTS_DIR / f"run_config_{run_id}.json"
    report_path = RESULTS_DIR / f"report_{run_id}.md"

    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path)
    significance.to_csv(significance_path, index=False)

    all_seasons = sorted(matches["season"].unique().tolist())
    run_config = {
        "run_id": run_id,
        "league": LEAGUE_CODE,
        "seasons_available": all_seasons,
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "evaluated_seasons": all_seasons[MIN_TRAINING_SEASONS:],
        "models": list(MODEL_BUILDERS.keys()),
        "hyperparameters": {
            "elo": {"k_factor": 20.0, "home_advantage": 100.0, "initial_rating": 1500.0},
            "dixon_coles": {"xi_per_day": 0.0065},
        },
    }
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    calibration_plot_paths = []
    for model_name in MODEL_BUILDERS:
        model_predictions = predictions[predictions["model"] == model_name]
        probs = model_predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = model_predictions["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            plot_path = RESULTS_DIR / f"calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, plot_path)
            calibration_plot_paths.append(plot_path.name)

    report = _build_report_markdown(run_config, summary, significance, recommendation, calibration_plot_paths)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")

    return summary


def _build_report_markdown(run_config, summary, significance, recommendation, calibration_plot_paths) -> str:
    lines = [
        f"# Research Benchmark Report - {run_config['run_id']}",
        "",
        f"League: {run_config['league']}. "
        f"Trained on seasons {run_config['seasons_available'][0]} onward; "
        f"evaluated (walk-forward, expanding window) on "
        f"{run_config['evaluated_seasons'][0]}-{run_config['evaluated_seasons'][-1]} "
        f"({len(run_config['evaluated_seasons'])} seasons, "
        f"{int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "Untuned hyperparameters used (literature defaults, not fit to this data - "
        "see Next Steps): "
        f"Elo K={run_config['hyperparameters']['elo']['k_factor']}, "
        f"home_advantage={run_config['hyperparameters']['elo']['home_advantage']}; "
        f"Dixon-Coles xi={run_config['hyperparameters']['dixon_coles']['xi_per_day']}/day.",
        "",
        "## Comparison",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "Lower is better for log loss, RPS, Brier score, and ECE (expected calibration error). "
        "Runtime is wall-clock seconds summed across all walk-forward folds on this machine - "
        "not comparable across hardware, only relative across these 4 models on this run.",
        "",
        "## Statistical significance (paired, per-match log loss)",
        "",
        significance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "p < 0.05 on both the paired t-test and Wilcoxon signed-rank test is treated as significant; "
        "the two are reported together since log-loss differences are often skewed by a small number "
        "of confidently-wrong predictions, which the t-test is more sensitive to than Wilcoxon.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Next steps",
        "",
        "Every model above uses literature-default hyperparameters, not values tuned to this dataset "
        "(Elo K-factor and home advantage; Dixon-Coles time-decay xi, rho initialization; the Poisson/"
        "Dixon-Coles goal-truncation limit; alternative time-decay windows). The next research "
        "milestone is a hyperparameter optimization pass over these, with the same walk-forward "
        "evaluation discipline used here, before any model is treated as a settled choice - a tuned "
        "Elo could plausibly beat an untuned Dixon-Coles, or vice versa, and that is exactly the kind "
        "of evidence this benchmark cannot yet rule out.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in calibration_plot_paths]

    return "\n".join(lines)


if __name__ == "__main__":
    main()
