"""Multi-league benchmark: does the champion generalize beyond England?

Everything so far was established on one league (the Premier League). Two
questions follow, and this benchmark answers both:

1. **Generalization.** Run the identical walk-forward benchmark independently on
   each of Europe's top five leagues. Does the ensemble still win? Does the
   ordering of the models hold? A champion that only wins in England is an
   overfitted champion.

2. **Statistical power.** The ensemble's margin over Elo was real but borderline
   on the EPL alone (~3,000 matches; significant on Wilcoxon, not on the paired
   t-test). Pooling five leagues gives ~14,000 matches on the same models, which
   is finally enough evidence to settle it. Pooled significance is computed on
   league-namespaced match keys so no two leagues' fixtures can collide.

Each league is trained and evaluated strictly within itself (no cross-league
training): a model fit on La Liga predicts La Liga. Pooling happens only at the
scoring stage, where it is legitimate - the same models, scored on more matches.

Output goes to research/results/ with a `multileague_` prefix.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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

LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
MIN_TRAINING_SEASONS = 4


def _ensemble():
    return EnsembleModel(
        {"elo": EloModel, "poisson_xg": PoissonXGModel, "dixon_coles_xg": DixonColesXGModel}
    )


MODEL_BUILDERS = {
    "baseline": BaselineFrequencyModel,
    "elo": EloModel,
    "poisson_xg": PoissonXGModel,
    "dixon_coles_xg": DixonColesXGModel,
    "ensemble": _ensemble,
}


def run_all_leagues():
    """Independent walk-forward per league. Returns (per_league_log_loss,
    pooled_predictions, pooled_runtimes)."""
    per_league = {}
    all_predictions = []
    pooled_runtimes = defaultdict(
        lambda: {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 0}
    )

    for league in LEAGUES:
        matches = load_understat_matches(league)
        logger.info("=== %s: %d matches ===", league, len(matches))
        predictions, runtimes = run_walk_forward(
            matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS
        )
        summary = summarize(predictions, runtimes)
        per_league[league] = summary["log_loss"]

        predictions["league"] = league
        all_predictions.append(predictions)
        for model, r in runtimes.items():
            for key in ("fit_seconds", "predict_seconds", "n_folds"):
                pooled_runtimes[model][key] += r[key]

    return (
        pd.DataFrame(per_league),  # index=model, columns=league
        pd.concat(all_predictions, ignore_index=True),
        dict(pooled_runtimes),
    )


def pooled_significance(pooled_predictions):
    """Namespace fixtures by league so identical (date, teams) keys cannot
    collide across leagues, then run the standard paired tests."""
    df = pooled_predictions.copy()
    df["home_team"] = df["league"] + ":" + df["home_team"]
    return pairwise_significance(df, list(MODEL_BUILDERS))


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
    return (
        f" ({'SIGNIFICANT' if sig else 'not significant'}: "
        f"paired t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"
    )


def recommend(per_league, pooled_summary, significance) -> str:
    winners = per_league.idxmin()  # best model per league
    ensemble_wins = int((winners == "ensemble").sum())
    lines = [
        f"**Generalization:** the ensemble is the best model in "
        f"**{ensemble_wins} of {len(LEAGUES)}** leagues.",
        "",
        "Winner per league: "
        + ", ".join(f"{lg} -> {winners[lg]}" for lg in per_league.columns)
        + ".",
        "",
        f"**Pooled (all {len(LEAGUES)} leagues, "
        f"{int(pooled_summary['n_predictions'].iloc[0])} matches per model):** "
        f"best is **{pooled_summary.sort_values('log_loss').index[0]}** "
        f"({pooled_summary['log_loss'].min():.4f}).",
        "",
        "Pooled head-to-heads (the point of pooling is statistical power the "
        "single-league test lacked):",
    ]
    for a, b in [("ensemble", "elo"), ("ensemble", "dixon_coles_xg"), ("elo", "dixon_coles_xg")]:
        if a in pooled_summary.index and b in pooled_summary.index:
            ll_a = pooled_summary.loc[a, "log_loss"]
            ll_b = pooled_summary.loc[b, "log_loss"]
            better = a if ll_a < ll_b else b
            lines.append(
                f"- **{a}** {ll_a:.4f} vs **{b}** {ll_b:.4f} -> {better} better"
                f"{_sig_text(significance, a, b)}."
            )

    if "ensemble" in pooled_summary.index and "elo" in pooled_summary.index:
        row = significance[
            ((significance["model_a"] == "ensemble") & (significance["model_b"] == "elo"))
            | ((significance["model_a"] == "elo") & (significance["model_b"] == "ensemble"))
        ]
        if not row.empty:
            p_t = float(row.iloc[0]["paired_t_pvalue"])
            p_w = float(row.iloc[0]["wilcoxon_pvalue"])
            beats = pooled_summary.loc["ensemble", "log_loss"] < pooled_summary.loc["elo", "log_loss"]
            if beats and p_t < 0.05 and p_w < 0.05:
                lines.append(
                    "\n**Verdict: the ensemble's win over Elo is now significant on "
                    "both tests.** With ~5x the matches, the margin that was "
                    "borderline on the EPL alone holds up. The ensemble is the "
                    "champion, and it generalizes."
                )
            elif beats:
                lines.append(
                    "\n**Verdict: the ensemble still leads Elo, but the margin "
                    "remains not fully significant even pooled.** Treat it as the "
                    "provisional champion; the two models are close."
                )
            else:
                lines.append(
                    "\n**Verdict: pooled across leagues, the ensemble does NOT beat "
                    "Elo.** The EPL-only lead did not generalize - a warning that "
                    "the champion was partly an artefact of one league."
                )
    return "\n".join(lines)


def _build_report(run_config, per_league, pooled_summary, significance, recommendation, plots) -> str:
    league_table = per_league.copy()
    league_table["pooled"] = pooled_summary["log_loss"]
    lines = [
        f"# Multi-League Benchmark - {run_config['run_id']}",
        "",
        f"Europe's top five leagues (Understat), each trained and evaluated "
        f"strictly within itself, walk-forward, {run_config['min_training_seasons']} "
        f"training-only seasons then 8 evaluated. Pooled scoring across "
        f"{int(pooled_summary['n_predictions'].iloc[0])} matches per model.",
        "",
        "**Questions:** (1) does the champion established on the Premier League "
        "generalize to other leagues, or was it an artefact of one competition? "
        "(2) Pooling ~5x the matches, is the ensemble's borderline win over Elo "
        "real?",
        "",
        "## Log loss by league (lower is better)",
        "",
        league_table.to_markdown(floatfmt=".4f"),
        "",
        "## Pooled comparison (all leagues)",
        "",
        pooled_summary.to_markdown(floatfmt=".4f"),
        "",
        "## Pooled statistical significance (paired, per-match log loss)",
        "",
        significance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Notes",
        "",
        "Each league is trained only on its own history - no cross-league "
        "transfer. Pooling happens at scoring time only. Match keys are "
        "namespaced by league so fixtures cannot collide in the paired tests.",
        "",
        "## Calibration plots (pooled)",
        "",
    ] + [f"- {name}" for name in plots]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_multileague.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    per_league, pooled_predictions, pooled_runtimes = run_all_leagues()
    pooled_summary = summarize(pooled_predictions, pooled_runtimes)
    significance = pooled_significance(pooled_predictions)
    recommendation = recommend(per_league, pooled_summary, significance)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pooled_predictions.to_csv(RESULTS_DIR / f"multileague_predictions_{run_id}.csv", index=False)
    per_league.to_csv(RESULTS_DIR / f"multileague_by_league_{run_id}.csv")
    pooled_summary.to_csv(RESULTS_DIR / f"multileague_pooled_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"multileague_significance_{run_id}.csv", index=False)

    run_config = {
        "run_id": run_id,
        "leagues": LEAGUES,
        "min_training_seasons": MIN_TRAINING_SEASONS,
        "models": list(MODEL_BUILDERS),
    }
    (RESULTS_DIR / f"multileague_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in MODEL_BUILDERS:
        mp = pooled_predictions[pooled_predictions["model"] == model_name]
        probs = mp[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = mp["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            path = RESULTS_DIR / f"multileague_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, path)
            plot_names.append(path.name)

    report = _build_report(
        run_config, per_league, pooled_summary, significance, recommendation, plot_names
    )
    (RESULTS_DIR / f"multileague_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return pooled_summary


if __name__ == "__main__":
    main()
