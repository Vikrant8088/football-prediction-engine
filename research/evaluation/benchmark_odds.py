"""Phase 4: the real finish line - can the engine beat the closing betting line?

The betting market aggregates everyone's information and money, and the CLOSING
line (Pinnacle's final price before kickoff) is the sharpest public forecast of
a football match that exists. It is the hardest honest benchmark available, and
the project's vision names it exactly that: odds are a yardstick, never an input
("copy bookmaker predictions" is an explicit non-goal).

This benchmark scores the market as if it were just another model, on the
identical matches our champion predicted, using the identical metrics. Three
things come out of it:

1. **How good is the engine, really?** One number, against the best public
   forecast rather than against our own weaker models.
2. **How much predictable signal is left?** The market's edge over us bounds
   what any further feature could possibly add.
3. **A progress metric:** what fraction of the naive-baseline -> market gap the
   engine has already captured.

That third number is the useful one for deciding what to build next. If we have
captured most of the achievable edge, hunting small features (weather, referees)
is provably near-pointless; if a large gap remains, it is worth hunting.

Only matches with usable Pinnacle closing odds are scored, and every model is
restricted to that same set, so all comparisons are paired and fair.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.odds_loader import load_closing_odds
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
FOOTBALL_DATA_LEAGUE = "E0"
MIN_TRAINING_SEASONS = 4
MARKET = "market_closing"


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


def _key(df):
    return list(zip(df["season"], df["home_team"], df["away_team"]))


def attach_market(predictions: pd.DataFrame, odds: pd.DataFrame):
    """Restrict every model to matches that have closing odds, and add the market
    as an extra 'model' scored on exactly those matches."""
    predictions = predictions.copy()
    odds = odds.copy()
    predictions["key"] = _key(predictions)
    odds["key"] = _key(odds)

    common = set(predictions["key"]) & set(odds["key"])
    predictions = predictions[predictions["key"].isin(common)]

    any_model = predictions["model"].iloc[0]
    fixtures = predictions[predictions["model"] == any_model][
        ["key", "date", "season", "home_team", "away_team", "result"]
    ]
    market = fixtures.merge(odds[["key", "p_home", "p_draw", "p_away"]], on="key")
    market["model"] = MARKET

    pooled = pd.concat([predictions, market], ignore_index=True, sort=False)
    return pooled.drop(columns=["key"]), len(common)


def _accuracy(group: pd.DataFrame) -> float:
    picks = np.array(["H", "D", "A"])[
        group[["p_home", "p_draw", "p_away"]].to_numpy().argmax(axis=1)
    ]
    return float((picks == group["result"].to_numpy()).mean())


def recommend(summary, significance, accuracy) -> str:
    ll_market = summary.loc[MARKET, "log_loss"]
    ll_base = summary.loc["baseline", "log_loss"]
    ll_eng = summary.loc["ensemble", "log_loss"]

    row = significance[
        ((significance["model_a"] == "ensemble") & (significance["model_b"] == MARKET))
        | ((significance["model_a"] == MARKET) & (significance["model_b"] == "ensemble"))
    ]
    sig_txt = ""
    if not row.empty:
        p_t = float(row.iloc[0]["paired_t_pvalue"])
        p_w = float(row.iloc[0]["wilcoxon_pvalue"])
        sig = p_t < 0.05 and p_w < 0.05
        sig_txt = (
            f" ({'significant' if sig else 'NOT significant'}: "
            f"paired t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"
        )

    # What fraction of the achievable edge (naive -> market) has the engine taken?
    captured = (ll_base - ll_eng) / (ll_base - ll_market) if ll_base > ll_market else float("nan")

    lines = [
        f"Closing line (Pinnacle): log loss **{ll_market:.4f}**. "
        f"Engine (ensemble): **{ll_eng:.4f}**. Naive baseline: {ll_base:.4f}.",
        "",
    ]
    if ll_eng < ll_market:
        lines.append(
            f"**The engine BEATS the closing line** by {ll_market - ll_eng:.4f} log "
            f"loss{sig_txt}. That is a remarkable result and should be treated with "
            f"suspicion until re-verified on other leagues and checked for any "
            f"lookahead in the odds join."
        )
    else:
        lines.append(
            f"**The closing line beats the engine** by {ll_eng - ll_market:.4f} log "
            f"loss{sig_txt}. This is the expected and honest outcome - the market is "
            f"the strongest public forecast there is."
        )
    lines += [
        "",
        f"**Progress metric — share of the achievable edge captured: "
        f"{100 * captured:.0f}%.** Moving from the naive baseline "
        f"({ll_base:.4f}) to the closing line ({ll_market:.4f}) is the entire "
        f"span of publicly-extractable skill; the engine has covered "
        f"{100 * captured:.0f}% of it.",
        "",
        "Top-pick accuracy on the same matches: "
        + ", ".join(f"{m} {100 * a:.1f}%" for m, a in accuracy.items())
        + ".",
        "",
        "**What this means for what to build next.** The remaining gap to the "
        "market bounds what ANY additional feature could add. A small residual "
        "gap means low-rated features (weather, referees) cannot plausibly pay "
        "for themselves; a large one means real signal is still on the table.",
    ]
    return "\n".join(lines)


def _build_report(run_config, summary, significance, recommendation, plots) -> str:
    return "\n".join([
        f"# Phase 4 Benchmark: vs the Closing Bookmaker Line - {run_config['run_id']}",
        "",
        f"League: {run_config['league']} (Understat matches + Pinnacle closing odds "
        f"from football-data.co.uk). Walk-forward, evaluated on "
        f"{int(summary['n_predictions'].iloc[0])} matches that have usable closing "
        f"odds ({run_config['seasons_scored']}).",
        "",
        "**Question:** can the engine beat the closing betting line - the sharpest "
        "public forecast of a football match? Odds are a yardstick, never a model "
        "input. The market is scored here as if it were just another model, on the "
        "identical matches and metrics.",
        "",
        "Bookmaker margin (overround) is removed by proportional normalisation; "
        f"mean overround was {100 * run_config['mean_overround']:.2f}%.",
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
        "## Caveats",
        "",
        "- Overround is removed proportionally; Shin's method or a power "
        "adjustment would slightly reduce the implied long-shot probabilities and "
        "could move the market's score a little.",
        "- The closing line is measured just before kickoff and therefore knows "
        "the confirmed line-ups and late team news. Our engine does not. Part of "
        "the residual gap is that information advantage, not modelling skill.",
        "- EPL only. The multi-league benchmark should be repeated against odds.",
        "",
        "## Calibration plots",
        "",
    ] + [f"- {name}" for name in plots])


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_odds.log",
        level="INFO",
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    predictions, runtimes = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)

    odds = load_closing_odds(FOOTBALL_DATA_LEAGUE)
    pooled, n_common = attach_market(predictions, odds)
    logger.info("Scoring %d matches that have closing odds", n_common)

    runtimes = dict(runtimes)
    runtimes[MARKET] = {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 1}

    summary = summarize(pooled, runtimes)
    all_models = list(MODEL_BUILDERS) + [MARKET]
    significance = pairwise_significance(pooled, all_models)
    accuracy = {m: _accuracy(g) for m, g in pooled.groupby("model")}
    recommendation = recommend(summary, significance, accuracy)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pooled.to_csv(RESULTS_DIR / f"odds_predictions_{run_id}.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"odds_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"odds_significance_{run_id}.csv", index=False)

    scored_seasons = sorted(pooled["season"].unique())
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "odds_source": "football-data.co.uk PSCH/PSCD/PSCA (Pinnacle closing)",
        "seasons_scored": f"{scored_seasons[0]} to {scored_seasons[-1]}",
        "n_matches": n_common,
        "mean_overround": float(odds["overround"].mean()),
        "models": all_models,
        "accuracy": accuracy,
    }
    (RESULTS_DIR / f"odds_run_config_{run_id}.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    plot_names = []
    for model_name in all_models:
        mp = pooled[pooled["model"] == model_name]
        probs = mp[["p_home", "p_draw", "p_away"]].to_numpy()
        outcomes = mp["result"].to_numpy()
        for outcome_label in ("H", "D", "A"):
            path = RESULTS_DIR / f"odds_calibration_{model_name}_{outcome_label}_{run_id}.png"
            plot_calibration_curve(probs, outcomes, outcome_label, path)
            plot_names.append(path.name)

    report = _build_report(run_config, summary, significance, recommendation, plot_names)
    (RESULTS_DIR / f"odds_report_{run_id}.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
