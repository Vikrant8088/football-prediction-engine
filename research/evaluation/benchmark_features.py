"""Phase 3a benchmark: does the "tiredness" signal (rest + fixture congestion)
actually improve prediction?

The clean, controlled experiment: fit one logistic model on FORM features only,
and an otherwise-identical one on FORM + TIREDNESS features. Both see the same
matches under the same walk-forward discipline, so any out-of-sample difference
is attributable to the tiredness features and nothing else - the same design
that isolated goals-vs-xG in Phase 1. The Elo model and the current champion
(the ensemble) are run alongside for context.

Feature models need feature-laden fixtures (rest/form are per-match numbers,
not derivable from team names), so this benchmark uses its own feature-aware
walk-forward loop rather than the goals-only harness. Features are precomputed
once on the full timeline - each match's features use only earlier matches (see
research/features/builders and its leakage test), and each model is still fit
only on seasons strictly before the one it predicts, so there is no leakage.

Output is written to research/results/ with a `features_` prefix.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR, summarize
from research.evaluation.calibration import plot_calibration_curve
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel
from research.experiments.ensemble import EnsembleModel
from research.experiments.feature_logistic import FeatureLogisticModel
from research.experiments.poisson_xg import PoissonXGModel
from research.features.builders import (
    ALL_FEATURES,
    FORM_FEATURES,
    add_match_features,
)

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
MIN_TRAINING_SEASONS = 4

# Controlled contrast: same model, form-only vs form+tiredness.
FEATURE_MODEL_SPECS = {
    "logistic_form": FORM_FEATURES,
    "logistic_form_rest": ALL_FEATURES,  # form + rest + congestion
}


def _ensemble_builder():
    return EnsembleModel(
        {"elo": EloModel, "poisson_xg": PoissonXGModel, "dixon_coles_xg": DixonColesXGModel}
    )


REFERENCE_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "ensemble": _ensemble_builder,
}


def run_feature_walk_forward(featured: pd.DataFrame):
    """Walk-forward over seasons. Feature models receive the full feature-laden
    test rows; reference models receive only (home_team, away_team) - each gets
    what its contract expects, all scored on the same matches."""
    seasons = sorted(featured["season"].unique())
    eval_seasons = seasons[MIN_TRAINING_SEASONS:]

    predictions = []
    runtimes = defaultdict(lambda: {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 0})

    for season in eval_seasons:
        train_df = featured[featured["season"] < season]
        test_df = featured[featured["season"] == season]

        builders = []
        builders += [
            (name, lambda cols=cols: FeatureLogisticModel(cols), test_df)
            for name, cols in FEATURE_MODEL_SPECS.items()
        ]
        builders += [
            (name, build, test_df[["home_team", "away_team"]])
            for name, build in REFERENCE_BUILDERS.items()
        ]

        for model_name, build_model, predict_input in builders:
            logger.info("Fitting %s, evaluating on %s", model_name, season)
            model = build_model()

            t0 = time.perf_counter()
            model.fit(train_df)
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            probs = model.predict_proba(predict_input)
            predict_s = time.perf_counter() - t0

            runtimes[model_name]["fit_seconds"] += fit_s
            runtimes[model_name]["predict_seconds"] += predict_s
            runtimes[model_name]["n_folds"] += 1

            fold = test_df[["date", "season", "home_team", "away_team", "result"]].copy()
            fold["model"] = model_name
            fold["p_home"] = probs[:, 0]
            fold["p_draw"] = probs[:, 1]
            fold["p_away"] = probs[:, 2]
            predictions.append(fold)

    return pd.concat(predictions, ignore_index=True), dict(runtimes)


def recommend(summary, significance) -> str:
    ranked = summary.sort_values("log_loss")
    lines = [f"Best model by log loss: **{ranked.index[0]}** ({ranked.iloc[0]['log_loss']:.4f})."]

    if "logistic_form" in summary.index and "logistic_form_rest" in summary.index:
        ll_form = summary.loc["logistic_form", "log_loss"]
        ll_rest = summary.loc["logistic_form_rest", "log_loss"]
        better = "helps" if ll_rest < ll_form else "does NOT help"
        row = significance[
            (
                (significance["model_a"] == "logistic_form")
                & (significance["model_b"] == "logistic_form_rest")
            )
            | (
                (significance["model_a"] == "logistic_form_rest")
                & (significance["model_b"] == "logistic_form")
            )
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
        lines.append(
            f"**Tiredness signal (controlled test): adding rest + congestion "
            f"{better}.** form-only log loss {ll_form:.4f} vs form+tiredness "
            f"{ll_rest:.4f}{sig_txt}."
        )

    if "ensemble" in summary.index:
        lines.append(
            f"For context, the current champion (ensemble) scores "
            f"{summary.loc['ensemble', 'log_loss']:.4f} - these form/tiredness "
            f"logistic models are diagnostic probes for the tiredness signal, "
            f"not yet tuned to beat the champion."
        )
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    lines = [
        f"# Phase 3a Research Benchmark (tiredness) - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat). Walk-forward on "
        f"{run_config['evaluated_seasons'][0]} to {run_config['evaluated_seasons'][-1]} "
        f"({len(run_config['evaluated_seasons'])} seasons, "
        f"{int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "**Question:** does the free 'tiredness' signal - days of rest and "
        "fixture congestion - improve prediction? Controlled test: an identical "
        "logistic model fit on FORM features only vs on FORM + TIREDNESS "
        "features. Any out-of-sample gap is the tiredness signal alone.",
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
        "The bigger Phase 3 signals - injuries, suspensions and confirmed "
        "line-ups - require an API-Football key and become additional builders "
        "in research/features, folded into the same logistic/ensemble machinery "
        "tested here. Form itself is also a candidate signal to add to the "
        "ensemble if it proves additive.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_features.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    featured = add_match_features(matches)
    logger.info("Loaded + featurized %d matches for Understat %s", len(featured), UNDERSTAT_LEAGUE)

    predictions, runtimes = run_feature_walk_forward(featured)
    summary = summarize(predictions, runtimes)
    all_models = list(FEATURE_MODEL_SPECS) + list(REFERENCE_BUILDERS)
    significance = pairwise_significance(predictions, all_models)
    recommendation = recommend(summary, significance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictions.to_csv(RESULTS_DIR / f"features_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"features_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"features_significance_{run_id}.csv", index=False)

    all_seasons = sorted(featured["season"].unique().tolist())
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "source": "understat",
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "evaluated_seasons": all_seasons[MIN_TRAINING_SEASONS:],
        "feature_model_specs": {k: v for k, v in FEATURE_MODEL_SPECS.items()},
        "reference_models": list(REFERENCE_BUILDERS),
    }
    (RESULTS_DIR / f"features_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in all_models:
        model_predictions = predictions[predictions["model"] == model_name]
        probs = model_predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = model_predictions["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            plot_path = RESULTS_DIR / f"features_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, plot_path)
            plot_names.append(plot_path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"features_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
