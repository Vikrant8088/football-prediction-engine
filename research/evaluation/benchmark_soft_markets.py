"""Phase 4d: can the engine beat a closing line where the market is SOFT?

Phase 4c settled it for the Premier League: no edge, at the open or the close.
Pinnacle prices the EPL with enormous care. It does not price League Two with
the same care - and the bookmaker tells us so itself, through its margin:

    Premier League  overround 2.38%      <- razor sharp, confident
    Championship              2.59%
    League One               3.11%
    League Two               3.15%      <- charging more because it is less sure

A wider margin is a bookmaker hedging against its own uncertainty. That is where
a public-data model has its only realistic chance of beating a closing line -
the same reason value investors do not hunt for bargains in Apple stock.

Design. Matches and odds come from the SAME football-data.co.uk file, so there
is no cross-source join and no team-name mapping at all. Understat does not
cover these divisions, so there is no xG: the engine here is goal-only. We use
**Elo** as the probe - it carries ~72% of the champion ensemble's weight and, on
the EPL, scores within 0.004 of the full ensemble - and it fits in 0.1s, which
makes a nine-league sweep cheap. The Premier League is included as the sharp-
market reference row, scored with the identical model, so the comparison across
leagues is apples-to-apples. What we are reading is the TREND in the gap.

If a league shows a small or negative gap, it earns a deeper run with the full
goal-only ensemble.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.loader import load_league_matches
from research.data.odds_loader import load_closing_odds
from research.evaluation.benchmark import RESULTS_DIR, run_walk_forward
from research.evaluation.metrics import evaluate_all
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.elo import EloModel

logger = logging.getLogger(__name__)

# name -> (football-data code); ordered sharp -> soft by observed overround.
LEAGUES = [
    ("Premier League", "E0"),
    ("Championship", "E1"),
    ("Scottish Prem", "SC0"),
    ("Serie B", "I2"),
    ("Bundesliga 2", "D2"),
    ("Segunda", "SP2"),
    ("League One", "E2"),
    ("Ligue 2", "F2"),
    ("League Two", "E3"),
]
MIN_TRAINING_SEASONS = 8
MODEL_BUILDERS = {"baseline": BaselineFrequencyModel, "elo": EloModel}
PROB_COLS = ["p_home", "p_draw", "p_away"]


def _accuracy(df: pd.DataFrame) -> float:
    picks = np.array(["H", "D", "A"])[df[PROB_COLS].to_numpy().argmax(axis=1)]
    return float((picks == df["result"].to_numpy()).mean())


def score_league(league_code: str) -> dict:
    """Walk-forward the engine on one league, then score it and the closing line
    on the identical matches (those that have odds)."""
    matches = load_league_matches(league_code)
    predictions, _ = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)
    odds = load_closing_odds(league_code, map_names=False)

    # The date is part of the key: some leagues (Scottish Premiership, which
    # splits mid-season) have a team host the same opponent twice in a season,
    # so (season, home, away) is NOT unique. Both sides come from the same
    # football-data file, so the dates line up exactly.
    key = lambda df: list(zip(df["date"], df["home_team"], df["away_team"]))
    predictions["key"] = key(predictions)
    odds["key"] = key(odds)
    if odds["key"].duplicated().any():
        raise ValueError(f"{league_code}: odds join key is not unique - would duplicate matches")

    common = set(predictions["key"]) & set(odds["key"])
    predictions = predictions[predictions["key"].isin(common)]
    if len(common) < 200:
        raise ValueError(f"{league_code}: only {len(common)} matches with odds")

    base = predictions[predictions["model"] == "elo"][
        ["key", "date", "season", "home_team", "away_team", "result"]
    ]
    market = base.merge(odds[["key"] + PROB_COLS], on="key")
    market["model"] = "market_closing"

    pooled = pd.concat([predictions, market], ignore_index=True, sort=False).drop(columns=["key"])

    scores = {}
    for model, group in pooled.groupby("model"):
        metrics = evaluate_all(group[PROB_COLS].to_numpy(), group["result"].to_numpy())
        scores[model] = {"log_loss": metrics["log_loss"], "accuracy": _accuracy(group)}

    significance = pairwise_significance(pooled, ["elo", "market_closing"])
    row = significance.iloc[0]

    gap = scores["elo"]["log_loss"] - scores["market_closing"]["log_loss"]
    span = scores["baseline"]["log_loss"] - scores["market_closing"]["log_loss"]
    return {
        "n_matches": len(common),
        "overround": float(odds["overround"].mean()),
        "baseline": scores["baseline"]["log_loss"],
        "elo": scores["elo"]["log_loss"],
        "market": scores["market_closing"]["log_loss"],
        "gap_to_market": gap,
        "edge_captured": (span - gap) / span if span > 0 else float("nan"),
        "elo_accuracy": scores["elo"]["accuracy"],
        "market_accuracy": scores["market_closing"]["accuracy"],
        "paired_t_pvalue": float(row["paired_t_pvalue"]),
        "wilcoxon_pvalue": float(row["wilcoxon_pvalue"]),
    }


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_soft_markets.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    rows = {}
    for name, code in LEAGUES:
        logger.info("=== %s (%s) ===", name, code)
        try:
            rows[name] = score_league(code)
        except Exception as exc:  # a league with too little odds coverage
            logger.warning("Skipping %s (%s): %s", name, code, exc)

    table = pd.DataFrame(rows).T.sort_values("gap_to_market")
    beaten = table[table["gap_to_market"] < 0]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / f"soft_markets_{run_id}.csv")

    display = table[[
        "n_matches", "overround", "baseline", "elo", "market",
        "gap_to_market", "edge_captured", "elo_accuracy", "market_accuracy",
        "paired_t_pvalue",
    ]].rename(columns={"elo": "engine (elo)", "market": "closing line"})

    if len(beaten):
        verdict = (
            "**The engine BEATS the closing line in: "
            + ", ".join(f"**{lg}** (gap {beaten.loc[lg, 'gap_to_market']:+.4f}, "
                        f"t p={beaten.loc[lg, 'paired_t_pvalue']:.4f})" for lg in beaten.index)
            + ".** Treat any result whose p-value is not significant as noise, and "
            "re-verify the winners with the full goal-only ensemble before believing them."
        )
    else:
        verdict = (
            "**The closing line beats the engine in every league tested.** Even where "
            "the bookmaker's own margin says it is unsure, its price is still sharper "
            "than a public-data model. There is no soft market here - at least not one "
            "reachable with goals-only Elo."
        )

    corr = float(np.corrcoef(table["overround"], table["gap_to_market"])[0, 1])

    report = "\n".join([
        f"# Soft-Market Benchmark (Phase 4d) - {run_id}",
        "",
        "Can the engine beat a closing line where the market is less efficient? "
        "Matches and Pinnacle closing odds come from the same football-data.co.uk "
        "file, so there is no cross-source join. Understat does not cover these "
        "divisions, so the engine is goal-only: **Elo** is the probe (it carries "
        "~72% of the champion ensemble's weight and, on the EPL, scores within "
        "0.004 of it). The Premier League is the sharp-market reference row, scored "
        "with the identical model.",
        "",
        "`gap_to_market` = engine log loss - closing-line log loss. **Negative means "
        "the engine wins.** `edge_captured` = share of the baseline->market span the "
        "engine covers.",
        "",
        "## Results (sorted: closest to beating the market first)",
        "",
        display.to_markdown(floatfmt=".4f"),
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"Correlation between the bookmaker's margin (overround) and our gap to its "
        f"price: **r = {corr:.3f}**. "
        + ("A negative correlation means: the wider the bookmaker's margin (the less "
           "sure it is), the closer we get - evidence that softness is real and "
           "exploitable in principle."
           if corr < -0.1 else
           "There is no clear relationship between the bookmaker's stated uncertainty "
           "and our ability to close on its price."),
        "",
        "## Caveats",
        "",
        "- Elo only. The full goal-only ensemble (Elo + Poisson + Dixon-Coles) would "
        "score a little better; any league that looks close here deserves that deeper run.",
        "- Beating a closing line on log loss is NOT the same as beating it after the "
        "bookmaker's margin. A profitable bet needs to beat the price, not the "
        "probability - the overround is the toll.",
        "- Odds are a yardstick, never a model input.",
    ])
    (RESULTS_DIR / f"soft_markets_report_{run_id}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return table


if __name__ == "__main__":
    main()
