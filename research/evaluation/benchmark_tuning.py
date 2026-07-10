"""Hyperparameter tuning benchmark: do fitted settings beat the textbook ones?

The Phase 0 report closed with an outstanding milestone: every model used
literature-default hyperparameters (Elo K=20, home advantage=100; Dixon-Coles
xi=0.0065/day), never fitted to this data. This benchmark settles it.

Each tuned model chooses its settings by nested walk-forward search - on every
fold, candidates are scored on an inner validation split of the TRAINING window
only, so the settings used to predict a season were never chosen by looking at
it (see research/experiments/tuning.py). Defaults and tuned variants are then
scored on identical matches, so any difference is the tuning and nothing else.

The headline question: does a tuned ensemble beat the reigning champion (the
default-hyperparameter ensemble, log loss 0.9925)?

Output goes to research/results/ with a `tuning_` prefix.
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
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel
from research.experiments.ensemble import EnsembleModel
from research.experiments.poisson_xg import PoissonXGModel
from research.experiments.tuning import build_tuned_dixon_coles_xg, build_tuned_elo

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
MIN_TRAINING_SEASONS = 4


def _ensemble_default():
    return EnsembleModel(
        {"elo": EloModel, "poisson_xg": PoissonXGModel, "dixon_coles_xg": DixonColesXGModel}
    )


def _ensemble_tuned():
    """Same blend, but Elo and Dixon-Coles-xG tune themselves inside each fold.
    Nesting is safe: the ensemble hands its bases an inner training split, and
    each base tunes on an inner split of THAT - never on the evaluated season."""
    return EnsembleModel(
        {
            "elo": build_tuned_elo,
            "poisson_xg": PoissonXGModel,
            "dixon_coles_xg": build_tuned_dixon_coles_xg,
        }
    )


MODEL_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "elo_tuned": build_tuned_elo,
    "dixon_coles_xg": DixonColesXGModel,
    "dixon_coles_xg_tuned": build_tuned_dixon_coles_xg,
    "ensemble": _ensemble_default,
    "ensemble_tuned": _ensemble_tuned,
}

PAIRS = [("elo", "elo_tuned"), ("dixon_coles_xg", "dixon_coles_xg_tuned"), ("ensemble", "ensemble_tuned")]


def _sig_text(significance, a, b) -> str:
    row = significance[
        ((significance["model_a"] == a) & (significance["model_b"] == b))
        | ((significance["model_a"] == b) & (significance["model_b"] == a))
    ]
    if row.empty:
        return ""
    p_t = float(row.iloc[0]["paired_t_pvalue"])
    p_w = float(row.iloc[0]["wilcoxon_pvalue"])
    sig = p_t < 0.05 and p_w < 0.05
    return f" ({'significant' if sig else 'not significant'}: t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"


def _final_params(matches):
    """Fit the tuners once on all-but-the-last season purely to report which
    settings they land on - the tuning result made inspectable."""
    seasons = sorted(matches["season"].unique())
    train = matches[matches["season"] < seasons[-1]]
    return {
        "elo": build_tuned_elo().fit(train).best_params_,
        "dixon_coles_xg": build_tuned_dixon_coles_xg().fit(train).best_params_,
    }


def recommend(summary, significance, final_params) -> str:
    ranked = summary.sort_values("log_loss")
    lines = [
        f"Best model by log loss: **{ranked.index[0]}** ({ranked.iloc[0]['log_loss']:.4f}).",
        "Settings chosen on the final training window: "
        + "; ".join(
            f"{m} -> " + ", ".join(f"{k}={v}" for k, v in p.items())
            for m, p in final_params.items()
        )
        + ".",
        "",
        "Default vs tuned, same model, identical matches:",
    ]
    for default, tuned in PAIRS:
        if default in summary.index and tuned in summary.index:
            ll_d = summary.loc[default, "log_loss"]
            ll_t = summary.loc[tuned, "log_loss"]
            verdict = "tuning HELPS" if ll_t < ll_d else "tuning does NOT help"
            lines.append(
                f"- **{default}**: {ll_d:.4f} -> tuned {ll_t:.4f} — **{verdict}**"
                f"{_sig_text(significance, default, tuned)}."
            )

    if "ensemble" in summary.index and "ensemble_tuned" in summary.index:
        ll_d = summary.loc["ensemble", "log_loss"]
        ll_t = summary.loc["ensemble_tuned", "log_loss"]
        if ll_t < ll_d:
            lines.append(
                f"\n**New champion: ensemble_tuned** ({ll_t:.4f}), beating the "
                f"default-hyperparameter ensemble ({ll_d:.4f}). Tuning was the "
                f"last untapped free gain and it paid."
            )
        else:
            lines.append(
                f"\nThe tuned ensemble ({ll_t:.4f}) does not beat the default one "
                f"({ll_d:.4f}). The literature defaults were already near-optimal "
                f"for this data - a genuinely useful null result: it closes the "
                f"outstanding Phase 0 milestone and rules tuning out as a lever."
            )
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    lines = [
        f"# Hyperparameter Tuning Benchmark - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat). Walk-forward on "
        f"{run_config['evaluated_seasons'][0]} to {run_config['evaluated_seasons'][-1]} "
        f"({len(run_config['evaluated_seasons'])} seasons, "
        f"{int(summary['n_predictions'].iloc[0])} matches per model).",
        "",
        "**Question:** the Phase 0 report's outstanding milestone - every model "
        "has used literature-default hyperparameters, never fitted to this data. "
        "Do tuned settings beat the textbook ones? Each tuned model picks its "
        "settings by nested walk-forward search on an inner validation split of "
        "the training window only, so no setting was ever chosen by looking at "
        "the season it predicts.",
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
        "## Search grids",
        "",
        f"```\n{json.dumps(run_config['grids'], indent=2)}\n```",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_tuning.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )
    from research.experiments.tuning import DIXON_COLES_XG_GRID, ELO_GRID

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    logger.info("Loaded %d matches", len(matches))

    predictions, runtimes = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)
    summary = summarize(predictions, runtimes)
    significance = pairwise_significance(predictions, list(MODEL_BUILDERS))
    final_params = _final_params(matches)
    recommendation = recommend(summary, significance, final_params)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    predictions.to_csv(RESULTS_DIR / f"tuning_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"tuning_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"tuning_significance_{run_id}.csv", index=False)

    all_seasons = sorted(matches["season"].unique().tolist())
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "evaluated_seasons": all_seasons[MIN_TRAINING_SEASONS:],
        "models": list(MODEL_BUILDERS),
        "grids": {"elo": ELO_GRID, "dixon_coles_xg": DIXON_COLES_XG_GRID},
        "final_window_params": final_params,
    }
    (RESULTS_DIR / f"tuning_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in MODEL_BUILDERS:
        mp = predictions[predictions["model"] == model_name]
        probs = mp[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = mp["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            path = RESULTS_DIR / f"tuning_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, path)
            plot_names.append(path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"tuning_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
