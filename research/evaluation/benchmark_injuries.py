"""Phase 3b benchmark: does knowing who is INJURED improve prediction?

The controlled experiment mirrors the tiredness test: an identical logistic
model fit on FORM features only vs on FORM + INJURY features (how many players
each side is missing). Any out-of-sample difference is the injury signal alone.

Data reality (free API plan): injuries cover only 2022/23-2024/25. So the
feature models train on the covered seasons available before each evaluated one
(2022/23 -> predict 2023/24; 2022/23+2023/24 -> predict 2024/25), giving TWO
evaluation seasons (~760 matches). That is a small sample - this is a first,
free read on whether the signal is worth paying to explore deeper, not a final
verdict. The strength models (Elo, ensemble) are trained on the FULL history
(2014-) and scored on the SAME evaluation matches, for honest context.

Output is written to research/results/ with an `injuries_` prefix.
"""

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.injury_loader import (
    COVERED_SEASONS,
    INJURY_FEATURES,
    INJURY_WEIGHT_FEATURES,
    add_injury_features,
)
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
from research.features.builders import FORM_FEATURES, add_match_features

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"

# Controlled 3-way contrast, identical model each time - only the features move:
#   none      -> form alone
#   count     -> + how MANY players are missing (Phase 3b; failed)
#   weighted  -> + how IMPORTANT the missing players are (Phase 3c)
FEATURE_MODEL_SPECS = {
    "logistic_form": FORM_FEATURES,
    "logistic_form_injury_count": FORM_FEATURES + INJURY_FEATURES,
    "logistic_form_injury_weight": FORM_FEATURES + INJURY_WEIGHT_FEATURES,
}


def _ensemble_builder():
    return EnsembleModel(
        {"elo": EloModel, "poisson_xg": PoissonXGModel, "dixon_coles_xg": DixonColesXGModel}
    )


STRENGTH_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "ensemble": _ensemble_builder,
}


def run_injury_walk_forward(featured: pd.DataFrame):
    """Feature models train on covered (injury-carrying) seasons before the
    evaluated one; strength models train on full history. Both score the same
    evaluation matches."""
    covered = sorted(COVERED_SEASONS)
    eval_seasons = covered[1:]  # first covered season has no covered predecessor

    predictions = []
    runtimes = defaultdict(lambda: {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 0})

    def record(name, test, probs):
        fold = test[["date", "season", "home_team", "away_team", "result"]].copy()
        fold["model"] = name
        fold["p_home"], fold["p_draw"], fold["p_away"] = probs[:, 0], probs[:, 1], probs[:, 2]
        predictions.append(fold)

    for season in eval_seasons:
        test = featured[featured["season"] == season]
        covered_train = featured[
            featured["season"].isin(COVERED_SEASONS) & (featured["season"] < season)
        ]
        full_train = featured[featured["season"] < season]

        for name, cols in FEATURE_MODEL_SPECS.items():
            t0 = time.perf_counter()
            model = FeatureLogisticModel(cols).fit(covered_train)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            probs = model.predict_proba(test)
            runtimes[name]["fit_seconds"] += fit_s
            runtimes[name]["predict_seconds"] += time.perf_counter() - t0
            runtimes[name]["n_folds"] += 1
            record(name, test, probs)

        fixtures = test[["home_team", "away_team"]]
        for name, build in STRENGTH_BUILDERS.items():
            t0 = time.perf_counter()
            model = build().fit(full_train)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            probs = model.predict_proba(fixtures)
            runtimes[name]["fit_seconds"] += fit_s
            runtimes[name]["predict_seconds"] += time.perf_counter() - t0
            runtimes[name]["n_folds"] += 1
            record(name, test, probs)

    return pd.concat(predictions, ignore_index=True), dict(runtimes)


def _sig_text(significance, model_a, model_b) -> str:
    row = significance[
        ((significance["model_a"] == model_a) & (significance["model_b"] == model_b))
        | ((significance["model_a"] == model_b) & (significance["model_b"] == model_a))
    ]
    if row.empty:
        return ""
    p_t = float(row.iloc[0]["paired_t_pvalue"])
    p_w = float(row.iloc[0]["wilcoxon_pvalue"])
    significant = p_t < 0.05 and p_w < 0.05
    return (
        f" ({'significant' if significant else 'not significant'}: "
        f"paired t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"
    )


def recommend(summary, significance) -> str:
    ranked = summary.sort_values("log_loss")
    lines = [f"Best model by log loss: **{ranked.index[0]}** ({ranked.iloc[0]['log_loss']:.4f})."]

    base = "logistic_form"
    variants = [
        ("logistic_form_injury_count", "how MANY players are missing (raw count)"),
        ("logistic_form_injury_weight", "how IMPORTANT the missing players are (weighted)"),
    ]
    if base in summary.index:
        ll_base = summary.loc[base, "log_loss"]
        lines.append(f"Baseline for the contrast: form-only log loss {ll_base:.4f}.")
        for model, description in variants:
            if model not in summary.index:
                continue
            ll = summary.loc[model, "log_loss"]
            verdict = "HELPS" if ll < ll_base else "does NOT help"
            lines.append(
                f"- Adding **{description}**: {ll:.4f} -> **{verdict}**"
                f"{_sig_text(significance, base, model)}."
            )

        both = {"logistic_form_injury_count", "logistic_form_injury_weight"} <= set(summary.index)
        if both:
            ll_c = summary.loc["logistic_form_injury_count", "log_loss"]
            ll_w = summary.loc["logistic_form_injury_weight", "log_loss"]
            better = "weighting by player importance" if ll_w < ll_c else "the raw count"
            lines.append(
                f"Head-to-head, **{better}** is the better of the two injury "
                f"encodings ({ll_w:.4f} weighted vs {ll_c:.4f} count)"
                f"{_sig_text(significance, 'logistic_form_injury_count', 'logistic_form_injury_weight')}."
            )

    lines.append(
        "Only ~760 evaluation matches (2 seasons - the free plan's injury "
        "window), so treat this as a first read, not a settled result. Importance "
        "is previous-season minutes, so a brand-new signing scores 0 even if he "
        "is a star - a known limitation of this proxy."
    )
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    lines = [
        f"# Phase 3b Research Benchmark (injuries) - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat matches + API-Football "
        f"injuries). Walk-forward on {', '.join(run_config['evaluated_seasons'])} "
        f"({int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "**Question:** does knowing who is unavailable improve prediction - and "
        "does it matter *how* we encode it? Controlled 3-way test with an "
        "identical logistic model: FORM alone, FORM + injury COUNT (how many are "
        "missing), and FORM + injury WEIGHT (how important the missing players "
        "are, by their previous-season minutes). Injury data (free API plan) "
        "covers 2022/23-2024/25 only, so this is a deliberately small first read.",
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
        "If injuries help: (1) upgrade to the paid API plan for ~6 seasons and "
        "re-test for significance; (2) weight injuries by player importance "
        "(minutes/market value) rather than a raw count; (3) add confirmed "
        "line-ups (who actually starts). If they do not help even here, record "
        "as rejected - the count-of-absences signal is too blunt without player "
        "importance.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_injuries.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    featured = add_injury_features(add_match_features(matches))
    predictions, runtimes = run_injury_walk_forward(featured)
    summary = summarize(predictions, runtimes)
    all_models = list(FEATURE_MODEL_SPECS) + list(STRENGTH_BUILDERS)
    significance = pairwise_significance(predictions, all_models)
    recommendation = recommend(summary, significance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictions.to_csv(RESULTS_DIR / f"injuries_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"injuries_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"injuries_significance_{run_id}.csv", index=False)

    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "evaluated_seasons": sorted(COVERED_SEASONS)[1:],
        "feature_model_specs": {k: v for k, v in FEATURE_MODEL_SPECS.items()},
        "strength_models": list(STRENGTH_BUILDERS),
    }
    (RESULTS_DIR / f"injuries_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in all_models:
        mp = predictions[predictions["model"] == model_name]
        probs = mp[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = mp["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            plot_path = RESULTS_DIR / f"injuries_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, plot_path)
            plot_names.append(plot_path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"injuries_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
