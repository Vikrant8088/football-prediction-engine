"""How often is the engine's predicted SCORE actually right?

The engine now emits a full scoreline distribution, so it can name a score. The
honest question is how often that score is correct - and, crucially, how that
compares to the ceiling any forecaster could reach.

Three numbers, all measured walk-forward on matches the model never saw:

  1. **Engine hit rate** - its single most likely score equals the real score.
  2. **Naive hit rate** - always predict 1-1, football's most common scoreline.
  3. **The ceiling** - the average probability the engine assigns to its own top
     score. For a well-calibrated forecaster this IS the expected hit rate: if
     the most likely score is only ~11% likely, you cannot be right much more
     than 11% of the time. Nobody can. Not Pinnacle, not anyone.

That third number is the point. Exact-score prediction is not a skill problem,
it is a variance problem: football simply does not concentrate enough
probability on any single scoreline. This benchmark exists so the engine's
score predictions are sold with their true hit rate attached, never as
certainties.

Also reported: top-3 hit rate, the log loss of the true scoreline under the full
grid (which measures the distribution, not the guess), and which scorelines the
engine actually names.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

LEAGUE = "EPL"
MIN_TRAINING_SEASONS = 4
NAIVE_SCORE = (1, 1)  # the most common football scoreline


def run():
    matches = load_understat_matches(LEAGUE)
    seasons = sorted(matches["season"].unique())

    rows = []
    for season in seasons[MIN_TRAINING_SEASONS:]:
        train = matches[matches["season"] < season]
        test = matches[matches["season"] == season]
        logger.info("Fitting on %d matches, scoring %s", len(train), season)
        model = ScorelineEnsemble().fit(train)

        for match in test.itertuples():
            grid = model.scoreline_grid(match.home_team, match.away_team)
            flat = np.argsort(grid, axis=None)[::-1]
            top3 = [np.unravel_index(i, grid.shape) for i in flat[:3]]
            best = top3[0]

            actual = (int(match.home_goals), int(match.away_goals))
            in_grid = actual[0] < grid.shape[0] and actual[1] < grid.shape[1]
            p_actual = float(grid[actual]) if in_grid else 0.0

            rows.append({
                "season": season,
                "actual": actual,
                "predicted": (int(best[0]), int(best[1])),
                "top_prob": float(grid[best]),
                "p_actual": p_actual,
                "hit": actual == (int(best[0]), int(best[1])),
                "hit_top3": any(actual == (int(h), int(a)) for h, a in top3),
                "naive_hit": actual == NAIVE_SCORE,
            })
    return pd.DataFrame(rows)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="research_benchmark_scorelines.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    df = run()

    n = len(df)
    engine = df["hit"].mean()
    top3 = df["hit_top3"].mean()
    naive = df["naive_hit"].mean()
    ceiling = df["top_prob"].mean()
    eps = 1e-12
    grid_log_loss = float(-np.log(np.clip(df["p_actual"], eps, 1.0)).mean())

    named = Counter(df["predicted"])
    actual_common = Counter(df["actual"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stats = {
        "run_id": run_id,
        "league": LEAGUE,
        "n_matches": int(n),
        "engine_exact_hit_rate": float(engine),
        "engine_top3_hit_rate": float(top3),
        "naive_always_1_1_hit_rate": float(naive),
        "theoretical_ceiling": float(ceiling),
        "scoreline_log_loss": grid_log_loss,
        "mean_prob_on_actual_score": float(df["p_actual"].mean()),
    }

    lines = [
        f"# Scoreline Prediction Benchmark - {run_id}",
        "",
        f"League: {LEAGUE}. Walk-forward on {n} matches the model never saw.",
        "",
        "**Question:** the engine can now name a score. How often is it right?",
        "",
        "## Results",
        "",
        "| | rate |",
        "|---|---|",
        f"| Engine's most likely score is exactly right | **{engine:.1%}** |",
        f"| The true score is in the engine's top 3 | **{top3:.1%}** |",
        f"| Naive: always predict 1-1 | {naive:.1%} |",
        f"| **Ceiling** (avg probability of the top score) | **{ceiling:.1%}** |",
        "",
        f"Log loss of the true scoreline under the full grid: {grid_log_loss:.4f}. "
        f"Mean probability the engine assigned to the score that actually "
        f"happened: {100 * stats['mean_prob_on_actual_score']:.1f}%.",
        "",
        "## Interpretation",
        "",
        f"The engine hits the exact score {engine:.1%} of the time, against a "
        f"ceiling of {ceiling:.1%}. It is therefore operating at roughly "
        f"**{100 * engine / ceiling:.0f}% of the maximum any forecaster could "
        f"achieve** - because the most likely scoreline in a football match is "
        f"itself only ~{ceiling:.0%} likely.",
        "",
        "Exact-score prediction is a **variance** problem, not a skill problem. "
        "No model, bookmaker or human beats this ceiling by much; the "
        "probability mass simply is not concentrated on one scoreline. The "
        "engine should therefore always publish the score WITH its probability "
        "attached, never as a certainty.",
        "",
        "## Scores the engine names most often",
        "",
        "| scoreline | engine named it | it actually happened |",
        "|---|---|---|",
    ]
    for score, count in named.most_common(6):
        lines.append(
            f"| {score[0]}-{score[1]} | {count} ({count / n:.1%}) | "
            f"{actual_common.get(score, 0)} ({actual_common.get(score, 0) / n:.1%}) |"
        )

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"scorelines_report_{run_id}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"scorelines_stats_{run_id}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(report)
    print(f"\nArtifacts written to {RESULTS_DIR}")
    return stats


if __name__ == "__main__":
    main()
