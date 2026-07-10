"""Does the engine hold any EDGE over the betting market?

Phase 4a established the honest scoreboard: Pinnacle's closing line beats our
champion (0.9464 vs 0.9821). That is expected - the closing price knows
everything we know, plus confirmed line-ups, late team news, and sharp money.
Beating it head-on is close to impossible for an information subset.

"Edge" is therefore not "beat the close on every match". It is one of these four
sharper, testable questions, all answerable with data already on disk:

  T1. **How much value arrives late?** Score the OPENING line against the
      CLOSING line. The gap is the worth of the information the market gains
      before kickoff - and therefore an upper bound on what team-news features
      could ever buy us.

  T2. **Can we beat the OPENING line?** A far more realistic target than the
      close, and the price a real bettor can actually take.

  T3. **Do we carry information the market does not fully price?** Blend our
      engine into the closing line with a walk-forward-fitted weight. If the
      blend beats the market alone AND the fitted weight on our engine is
      meaningfully positive, then our forecast contains signal the market
      missed. This is the cleanest scientific proof of edge there is.

  T4. **Closing Line Value (CLV).** When we disagree with the opening price,
      does the line subsequently move TOWARD us? Anticipating the market's own
      correction is the professional gold standard for a genuine edge, because
      it means you can take the open price and watch it become the right one.

INTEGRITY NOTE: the blend in T3 is a MEASUREMENT INSTRUMENT, not a shipped
model. Odds remain a yardstick, never an input to the champion - the project's
vision names "copy bookmaker predictions" an explicit non-goal. We fit the blend
only to ask what our engine adds, then we throw it away.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from research.data.odds_loader import load_pinnacle_odds
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR, run_walk_forward, summarize
from research.evaluation.metrics import log_loss
from research.evaluation.significance import pairwise_significance
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.elo import EloModel
from research.experiments.ensemble import EnsembleModel
from research.experiments.poisson_xg import PoissonXGModel

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"
FOOTBALL_DATA_LEAGUE = "E0"
MIN_TRAINING_SEASONS = 4
PROB_COLS = ["p_home", "p_draw", "p_away"]


def _ensemble():
    return EnsembleModel(
        {"elo": EloModel, "poisson_xg": PoissonXGModel, "dixon_coles_xg": DixonColesXGModel}
    )


MODEL_BUILDERS = {"ensemble": _ensemble}


def fit_blend_weight(engine: np.ndarray, market: np.ndarray, outcomes) -> float:
    """Weight w in [0,1] on the ENGINE in a linear pool w*engine + (1-w)*market,
    chosen to minimise log loss. w=0 means the engine adds nothing."""
    best_w, best_loss = 0.0, np.inf
    for w in np.linspace(0.0, 1.0, 101):
        blended = w * engine + (1.0 - w) * market
        loss = log_loss(blended, outcomes)
        if loss < best_loss:
            best_w, best_loss = float(w), loss
    return best_w


def build_frames(predictions: pd.DataFrame, odds: pd.DataFrame):
    """Restrict to matches with open+close odds; return a long frame carrying
    engine, market_open and market_close as three 'models'."""
    key = lambda df: list(zip(df["season"], df["home_team"], df["away_team"]))
    predictions = predictions.copy()
    odds = odds.copy()
    predictions["key"] = key(predictions)
    odds["key"] = key(odds)

    common = set(predictions["key"]) & set(odds["key"])
    engine = predictions[predictions["key"].isin(common)].copy()

    base = engine[["key", "date", "season", "home_team", "away_team", "result"]]
    merged = base.merge(odds, on="key", suffixes=("", "_odds"))

    def market_frame(prefix, name):
        f = base.merge(
            odds[["key", f"{prefix}_p_home", f"{prefix}_p_draw", f"{prefix}_p_away"]], on="key"
        ).rename(columns={
            f"{prefix}_p_home": "p_home", f"{prefix}_p_draw": "p_draw", f"{prefix}_p_away": "p_away",
        })
        f["model"] = name
        return f

    pooled = pd.concat(
        [engine, market_frame("open", "market_open"), market_frame("close", "market_close")],
        ignore_index=True, sort=False,
    )
    return pooled.drop(columns=["key"]), merged


def walk_forward_blend(engine_df: pd.DataFrame, merged: pd.DataFrame):
    """T3: for each evaluated season, fit the blend weight on EARLIER seasons
    only, then apply it to that season. No leakage; the first season, having no
    history, falls back to w=0 (pure market)."""
    df = engine_df.merge(
        merged[["season", "home_team", "away_team", "close_p_home", "close_p_draw", "close_p_away"]],
        on=["season", "home_team", "away_team"],
    )
    seasons = sorted(df["season"].unique())

    rows, weights = [], {}
    for i, season in enumerate(seasons):
        prior = df[df["season"].isin(seasons[:i])]
        if len(prior) == 0:
            w = 0.0
        else:
            w = fit_blend_weight(
                prior[PROB_COLS].to_numpy(),
                prior[["close_p_home", "close_p_draw", "close_p_away"]].to_numpy(),
                prior["result"].to_numpy(),
            )
        weights[season] = w

        cur = df[df["season"] == season]
        blended = w * cur[PROB_COLS].to_numpy() + (1 - w) * cur[
            ["close_p_home", "close_p_draw", "close_p_away"]
        ].to_numpy()
        out = cur[["date", "season", "home_team", "away_team", "result"]].copy()
        out[PROB_COLS] = blended
        out["model"] = "blend_engine_plus_market"
        rows.append(out)

    return pd.concat(rows, ignore_index=True), weights


def closing_line_value(merged: pd.DataFrame, engine_df: pd.DataFrame) -> dict:
    """T4: when the engine disagrees with the OPEN, does the line move toward it?

    Stacks all three outcomes. `disagreement` = engine - open; `movement` =
    close - open. A positive correlation means the market later moved the way we
    already were - i.e. we anticipated the market's own correction.
    """
    df = engine_df.merge(
        merged[[
            "season", "home_team", "away_team",
            "open_p_home", "open_p_draw", "open_p_away",
            "close_p_home", "close_p_draw", "close_p_away",
        ]],
        on=["season", "home_team", "away_team"],
    )
    engine = df[PROB_COLS].to_numpy()
    opening = df[["open_p_home", "open_p_draw", "open_p_away"]].to_numpy()
    closing = df[["close_p_home", "close_p_draw", "close_p_away"]].to_numpy()

    disagreement = (engine - opening).ravel()
    movement = (closing - opening).ravel()

    r, p = stats.pearsonr(disagreement, movement)
    strong = np.abs(disagreement) > 0.02
    same_direction = float(
        (np.sign(disagreement[strong]) == np.sign(movement[strong])).mean()
    )
    return {
        "n_matches": int(len(df)),
        "correlation": float(r),
        "p_value": float(p),
        "share_line_moved_toward_us": same_direction,
        "n_strong_disagreements": int(strong.sum()),
        "mean_abs_movement": float(np.abs(movement).mean()),
    }


def recommend(summary, significance, weights, clv) -> str:
    ll = summary["log_loss"]
    lines = []

    # T1
    lines.append(
        f"**T1 - value of late information.** Opening line {ll['market_open']:.4f} -> "
        f"closing line {ll['market_close']:.4f}. The market improves by "
        f"{ll['market_open'] - ll['market_close']:.4f} between the price going up and "
        f"kickoff. That is what team news + sharp money are worth, and it bounds "
        f"what any team-news feature could buy us."
    )
    # T2
    beat_open = ll["ensemble"] < ll["market_open"]
    lines.append(
        f"\n**T2 - can we beat the OPENING line?** engine {ll['ensemble']:.4f} vs "
        f"opening {ll['market_open']:.4f} -> "
        + ("**YES, the engine beats the opening price.**" if beat_open
           else "no, the opening price is already sharper than us.")
    )
    # T3
    blend_ll = ll.get("blend_engine_plus_market", float("nan"))
    mean_w = float(np.mean([w for w in weights.values()]))
    improves = blend_ll < ll["market_close"]
    row = significance[
        ((significance["model_a"] == "blend_engine_plus_market") & (significance["model_b"] == "market_close"))
        | ((significance["model_a"] == "market_close") & (significance["model_b"] == "blend_engine_plus_market"))
    ]
    sig = ""
    if not row.empty:
        p_t = float(row.iloc[0]["paired_t_pvalue"])
        p_w = float(row.iloc[0]["wilcoxon_pvalue"])
        sig = (
            f" ({'SIGNIFICANT' if (p_t < 0.05 and p_w < 0.05) else 'not significant'}: "
            f"t p={p_t:.4f}, Wilcoxon p={p_w:.4f})"
        )
    lines.append(
        f"\n**T3 - do we add information the market missed?** Blending the engine "
        f"into the closing line scores {blend_ll:.4f} vs the market's "
        f"{ll['market_close']:.4f}{sig}. Average fitted weight on our engine: "
        f"**{mean_w:.0%}** (per season: "
        + ", ".join(f"{s}={w:.0%}" for s, w in sorted(weights.items())) + ")."
    )
    if improves and mean_w > 0.01:
        lines.append(
            "  -> The market's own price is IMPROVED by listening to us. Our engine "
            "carries signal Pinnacle does not fully price. That is a real, if small, edge."
        )
    else:
        lines.append(
            "  -> The blend does not improve on the market. Our forecast appears to be "
            "a strict information subset of the closing line - no exploitable edge here."
        )
    # T4
    lines.append(
        f"\n**T4 - Closing Line Value.** Correlation between our disagreement with the "
        f"open and the line's subsequent movement: **r={clv['correlation']:.3f}** "
        f"(p={clv['p_value']:.2g}). On the {clv['n_strong_disagreements']} strong "
        f"disagreements, the line moved toward us "
        f"**{100 * clv['share_line_moved_toward_us']:.1f}%** of the time."
    )
    if clv["correlation"] > 0.05 and clv["p_value"] < 0.05:
        lines.append(
            "  -> The market tends to move OUR way. We anticipate its correction, which "
            "is the professional definition of an edge: take the opening price and the "
            "closing price validates you."
        )
    else:
        lines.append(
            "  -> No meaningful anticipation of the market's move."
        )
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_market_edge.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    predictions, _ = run_walk_forward(matches, MODEL_BUILDERS, MIN_TRAINING_SEASONS)
    odds = load_pinnacle_odds(FOOTBALL_DATA_LEAGUE)

    pooled, merged = build_frames(predictions, odds)
    engine_df = pooled[pooled["model"] == "ensemble"]

    blend, weights = walk_forward_blend(engine_df, merged)
    clv = closing_line_value(merged, engine_df)

    pooled = pd.concat([pooled, blend], ignore_index=True, sort=False)
    models = ["ensemble", "market_open", "market_close", "blend_engine_plus_market"]
    runtimes = {m: {"fit_seconds": 0.0, "predict_seconds": 0.0, "n_folds": 1} for m in models}

    summary = summarize(pooled, runtimes)
    significance = pairwise_significance(pooled, models)
    recommendation = recommend(summary, significance, weights, clv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary.to_csv(RESULTS_DIR / f"market_edge_summary_{run_id}.csv")
    significance.to_csv(RESULTS_DIR / f"market_edge_significance_{run_id}.csv", index=False)

    report = "\n".join([
        f"# Market Edge Benchmark - {run_id}",
        "",
        f"League: {UNDERSTAT_LEAGUE}. {int(summary['n_predictions'].iloc[0])} matches with both "
        f"Pinnacle OPENING and CLOSING odds.",
        "",
        "Four questions: what is late information worth (T1); can we beat the opening "
        "line (T2); do we carry signal the closing line misses (T3); and does the line "
        "move toward us when we disagree (T4 — closing line value)?",
        "",
        "> **Integrity note:** the T3 blend is a measurement instrument, not a shipped "
        "model. Odds remain a yardstick, never an input to the champion.",
        "",
        "## Comparison",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "## Significance (paired, per-match log loss)",
        "",
        significance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Findings",
        "",
        recommendation,
        "",
        "## Closing Line Value detail",
        "",
        f"```\n{json.dumps(clv, indent=2)}\n```",
    ])
    (RESULTS_DIR / f"market_edge_report_{run_id}.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
