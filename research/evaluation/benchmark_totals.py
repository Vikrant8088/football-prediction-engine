"""The last untested market: Over/Under 2.5 goals.

Every edge test so far has been on Win/Draw/Loss - a market where Elo, a pure
ratings model, carried ~72% of the champion's weight. But the xG models predict
*goals* directly. That is their core competence, and it has never once been
tested against the goals market. Pinnacle also charges a wider margin here
(~3.2% vs ~2.4% on 1X2), which is a bookmaker admitting it is less sure.

So the 1X2 result does NOT automatically transfer. This benchmark settles it.

Two separate questions, deliberately kept apart, because they are not the same:

  Q1. **Is our probability better than the market's?**  Scored by log loss and
      Brier on the binary over/under outcome, paired on identical matches.

  Q2. **Could it actually make money?**  A far higher bar. Profit requires
      beating the PRICE, not the probability - the bookmaker's margin is a toll
      you pay on every bet. We simulate flat 1-unit bets wherever the model sees
      positive expected value (p_model x odds > 1), at two price levels:
        - Pinnacle's closing price (margin ~3.2%): beating the sharpest book.
        - The best closing price across all books (margin ~0.16%): what a bettor
          shopping around would really get. This is the realistic test.

Three model grids are compared, because the rescaling matters here: the
ScorelineEnsemble forces its 1X2 marginals to the champion's, which reshapes the
totals distribution - possibly for the worse. So the raw Dixon-Coles-xG and
Poisson-xG grids are scored alongside it.

CAVEAT ON THE SIMULATION: betting the best closing price across books is
optimistic. Those prices are not always reachable, and accounts that win get
limited. Treat any positive ROI as an upper bound, and demand statistical
significance before believing it.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from research.data.odds_loader import load_totals_odds
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

LEAGUE = "EPL"
MIN_TRAINING_SEASONS = 4
LINE = 2.5
EDGE_THRESHOLD = 0.02  # only bet when the model sees >2% expected value
EPS = 1e-12


def _p_over(grid: np.ndarray, line: float = LINE) -> float:
    goals = np.arange(grid.shape[0])
    totals = goals[:, None] + goals[None, :]
    normalised = grid / grid.sum()
    return float(normalised[totals > line].sum())


def run_walk_forward() -> pd.DataFrame:
    matches = load_understat_matches(LEAGUE)
    seasons = sorted(matches["season"].unique())

    rows = []
    for season in seasons[MIN_TRAINING_SEASONS:]:
        train = matches[matches["season"] < season]
        test = matches[matches["season"] == season]
        logger.info("Fitting on %d matches, scoring %s", len(train), season)

        model = ScorelineEnsemble().fit(train)
        bases = model.base_models
        base_rate = float(((train["home_goals"] + train["away_goals"]) > LINE).mean())

        for match in test.itertuples():
            home, away = match.home_team, match.away_team
            rows.append({
                "season": season,
                "date": match.date,
                "home_team": home,
                "away_team": away,
                "over": bool((match.home_goals + match.away_goals) > LINE),
                "ensemble": _p_over(model.scoreline_grid(home, away)),
                "dixon_coles_xg": _p_over(bases["dixon_coles_xg"].score_grid(home, away)),
                "poisson_xg": _p_over(bases["poisson_xg"].score_grid(home, away)),
                "baseline": base_rate,
            })
    return pd.DataFrame(rows)


def _binary_log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def simulate_betting(df: pd.DataFrame, model: str, price: str) -> dict:
    """Flat 1-unit bets wherever the model sees positive expected value."""
    p_over = df[model].to_numpy()
    over = df["over"].to_numpy().astype(bool)
    odds_over = df[f"{price}_odds_over"].to_numpy()
    odds_under = df[f"{price}_odds_under"].to_numpy()

    ev_over = p_over * odds_over - 1.0
    ev_under = (1.0 - p_over) * odds_under - 1.0

    bet_over = ev_over > EDGE_THRESHOLD
    bet_under = (ev_under > EDGE_THRESHOLD) & ~bet_over

    profits = []
    for i in range(len(df)):
        if bet_over[i]:
            profits.append(odds_over[i] - 1.0 if over[i] else -1.0)
        elif bet_under[i]:
            profits.append(odds_under[i] - 1.0 if not over[i] else -1.0)
    profits = np.array(profits)

    if len(profits) < 2:
        return {"n_bets": int(len(profits)), "roi": float("nan"), "p_value": float("nan")}

    # Is mean profit per bet significantly different from zero?
    t_stat, p_value = stats.ttest_1samp(profits, 0.0)
    return {
        "n_bets": int(len(profits)),
        "bet_rate": float(len(profits) / len(df)),
        "total_profit_units": float(profits.sum()),
        "roi": float(profits.mean()),
        "p_value": float(p_value),
        "significant_profit": bool(p_value < 0.05 and profits.mean() > 0),
    }


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_totals.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    preds = run_walk_forward()
    odds = load_totals_odds("E0")

    df = preds.merge(
        odds, on=["season", "home_team", "away_team"], suffixes=("", "_odds")
    )
    df = df.rename(columns={"p_over": "market_closing"})
    logger.info("Scoring %d matches with closing over/under odds", len(df))

    models = ["ensemble", "dixon_coles_xg", "poisson_xg", "baseline", "market_closing"]
    y = df["over"].to_numpy().astype(float)

    scores = {}
    losses = {}
    for m in models:
        p = df[m].to_numpy()
        losses[m] = _binary_log_loss(p, y)
        scores[m] = {
            "log_loss": float(losses[m].mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
            "mean_p_over": float(p.mean()),
        }
    summary = pd.DataFrame(scores).T.sort_values("log_loss")

    # Paired significance vs the market, on identical matches.
    sig = {}
    for m in models:
        if m == "market_closing":
            continue
        diff = losses[m] - losses["market_closing"]
        t_p = float(stats.ttest_rel(losses[m], losses["market_closing"])[1])
        w_p = float(stats.wilcoxon(losses[m], losses["market_closing"])[1])
        sig[m] = {"mean_log_loss_diff": float(diff.mean()), "t_p": t_p, "wilcoxon_p": w_p}

    betting = {}
    for m in ["ensemble", "dixon_coles_xg", "poisson_xg"]:
        for price in ["pin", "best"]:
            betting[f"{m} @ {price}"] = simulate_betting(df, m, price)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    best_model = summary.index[0]
    market_ll = summary.loc["market_closing", "log_loss"]
    engine_ll = summary.loc["ensemble", "log_loss"]
    best_engine = min(
        ["ensemble", "dixon_coles_xg", "poisson_xg"], key=lambda m: scores[m]["log_loss"]
    )

    any_profit = [k for k, v in betting.items() if v.get("significant_profit")]

    lines = [
        f"# Over/Under 2.5 Goals Benchmark - {run_id}",
        "",
        f"League: {LEAGUE}. {len(df)} matches with Pinnacle closing over/under odds "
        f"({df['season'].min()} to {df['season'].max()}).",
        "",
        "The goals market is the one our xG models were actually built for, and it "
        "has never been tested. Pinnacle charges a wider margin here "
        f"({100 * df['overround'].mean():.2f}%) than on 1X2 (~2.4%).",
        "",
        "## Q1 - is our probability better than the market's?",
        "",
        summary.to_markdown(floatfmt=".4f"),
        "",
        "Paired vs the closing line (negative diff = we are better):",
        "",
        "| model | mean log-loss diff | paired t p | Wilcoxon p |",
        "|---|---|---|---|",
    ]
    for m, s in sig.items():
        lines.append(
            f"| {m} | {s['mean_log_loss_diff']:+.4f} | {s['t_p']:.4f} | {s['wilcoxon_p']:.4f} |"
        )

    lines += [
        "",
        f"Best forecaster: **{best_model}**. Our best model is **{best_engine}** "
        f"({scores[best_engine]['log_loss']:.4f}) vs the market's {market_ll:.4f} - "
        + ("**we are ahead**." if scores[best_engine]["log_loss"] < market_ll
           else f"the market is ahead by {scores[best_engine]['log_loss'] - market_ll:.4f}."),
        "",
        "## Q2 - could it actually make money?",
        "",
        "Flat 1-unit bets wherever the model sees >2% expected value. ROI is profit "
        "per unit staked. `pin` = Pinnacle's closing price; `best` = the best "
        "closing price across all books (what a bettor shopping around gets).",
        "",
        "| model @ price | bets | bet rate | ROI | p-value | profitable? |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in betting.items():
        if b["n_bets"] < 2:
            lines.append(f"| {name} | {b['n_bets']} | - | - | - | no bets |")
            continue
        lines.append(
            f"| {name} | {b['n_bets']} | {b['bet_rate']:.1%} | {b['roi']:+.2%} | "
            f"{b['p_value']:.4f} | {'**YES**' if b['significant_profit'] else 'no'} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if any_profit:
        lines.append(
            "**A statistically significant profit appears at: "
            + ", ".join(any_profit)
            + ".** Treat this with deep suspicion until it survives: other leagues, "
            "a held-out period, and the fact that best-available prices are not "
            "always reachable and winning accounts get limited."
        )
    else:
        lines.append(
            "**No model turns a statistically significant profit at any price level.** "
            "Even where the engine's probability is competitive, the bookmaker's "
            "margin is a toll that swallows the difference. The goals market is "
            "closed too."
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- Betting the best closing price across books is optimistic: those prices "
        "are not always reachable, and accounts that win get limited.",
        "- Overround removed by proportional normalisation.",
        "- EPL only; the sharpest league. Odds are a yardstick, never a model input.",
    ]

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"totals_report_{run_id}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"totals_stats_{run_id}.json").write_text(
        json.dumps({"scores": scores, "significance": sig, "betting": betting}, indent=2),
        encoding="utf-8",
    )
    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    main()
