"""The minutes ceiling, fast: how much could PERFECT lineup knowledge ever add?

Same question as `benchmark_fpl_minutes_ceiling.py` — give the minutes model perfect
foreknowledge of exactly how long every player actually played, and measure how much
the edge grows over the shipped recent-form model — but scored on the **unbudgeted
greedy legal XI** instead of the £100m squad.

Why this exists: the budgeted version is the decision-relevant one, but it is
pathologically slow. Perfect-minutes projections are finely-spaced continuous values
with almost no ties, which is precisely the case the branch-and-bound squad solver
handles worst; a full run burned 6+ CPU-hours over ~1,000 solves without finishing.
The greedy legal XI needs no knapsack, so it answers the same question in seconds.

What it is and is not:
  - it IS a directional upper bound: perfect minutes leak the gameweek's actual
    minutes, so any real lineup signal is strictly worse;
  - it is NOT the pre-registered £100m + captain figure. Budget constraints would
    likely compress the headroom, so treat the magnitude as indicative and the
    direction as the finding.

Both prediction frames are read from cache — this script never refits anything.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_warehouse.config.loader import load_config
from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.scorer import paired_summary
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_projections import _top11

logger = logging.getLogger(__name__)

CHAMPION_FRAME = "fpl_backtest_predictions_understat_min_hl2.csv"   # recent-form (shipped)
PERFECT_FRAME = "fpl_backtest_predictions_understat_perfect.csv"    # leaks actual minutes
BASELINE = "player_ppg"


def load_frames():
    processed = load_config().raw_data_dir.parent / "processed"
    for name in (CHAMPION_FRAME, PERFECT_FRAME):
        if not (processed / name).exists():
            raise SystemExit(
                "missing cached frame %s — run benchmark_fpl_minutes_ceiling.py far "
                "enough to build it, or regenerate via cached_predictions()." % name)
    return (pd.read_csv(processed / CHAMPION_FRAME),
            pd.read_csv(processed / PERFECT_FRAME))


def build_table(champion: pd.DataFrame, perfect: pd.DataFrame) -> pd.DataFrame:
    """Actual points of each picker's greedy legal XI, per season-gameweek."""
    return pd.DataFrame({
        "perfect": _top11(perfect, "ours"),
        "champion": _top11(champion, "ours"),
        "baseline": _top11(champion, BASELINE),   # identical in both frames
    }).dropna()


def build_report(table: pd.DataFrame, run_id: str) -> str:
    headroom = paired_summary(list(table["perfect"]), list(table["champion"]))
    champ_edge = paired_summary(list(table["champion"]), list(table["baseline"]))
    perfect_edge = paired_summary(list(table["perfect"]), list(table["baseline"]))

    lines = [
        "# The minutes ceiling (fast, unbudgeted legal XI) — " + run_id,
        "",
        "Perfect minutes LEAK each gameweek's actual minutes: an upper bound on any "
        "lineup/minutes signal, not a shippable model. Scored on the greedy legal XI "
        "(no budget, no captain) because the £100m squad solve is pathologically slow "
        "on finely-spaced perfect-minutes projections. **Directional, not the "
        "pre-registered £100m + captain figure.**",
        "",
        "%d gameweeks, %d seasons." % (len(table), table.index.get_level_values(0).nunique()),
        "",
        "| picker | actual pts/GW | edge vs `%s` |" % BASELINE,
        "|---|---|---|",
        "| `%s` baseline | %.2f | — |" % (BASELINE, table["baseline"].mean()),
        "| recent-form minutes (shipped champion) | %.2f | **%+.2f/GW** |" % (
            table["champion"].mean(), champ_edge["mean_gain_per_gw"]),
        "| PERFECT minutes (ceiling) | %.2f | **%+.2f/GW** |" % (
            table["perfect"].mean(), perfect_edge["mean_gain_per_gw"]),
        "",
        "## The headroom (perfect − recent-form)",
        "",
        "> **%+.3f pts/GW** (t p=%.4f, Wilcoxon p=%.4f), won %d/%d gameweeks." % (
            headroom["mean_gain_per_gw"], headroom["paired_t_p"], headroom["wilcoxon_p"],
            headroom["gameweeks_won"], headroom["gameweeks"]),
        "",
        "## Per-season headroom",
        "",
        "| season | perfect − recent-form (pts/GW) | GWs won |",
        "|---|---|---|",
    ]
    per_season = {}
    for season, group in table.groupby(level=0):
        s = paired_summary(list(group["perfect"]), list(group["champion"]))
        per_season[season] = s
        lines.append("| %s | %+.3f | %d/%d |" % (
            season, s["mean_gain_per_gw"], s["gameweeks_won"], s["gameweeks"]))

    positive = sum(1 for s in per_season.values() if s["mean_gain_per_gw"] > 0)
    lines += [
        "",
        "## Reading it",
        "",
        "This headroom is the MAXIMUM any minutes/lineup data could add on top of the "
        "recent-form model. Live predicted lineups are imperfect, so they capture only "
        "a FRACTION of it.",
        "",
        "**Positive in %d/%d seasons.**" % (positive, len(per_season)),
        "",
        "**Important caveat — the live system is already better informed than this "
        "champion.** The backtest cannot see FPL's injury flags (they are published "
        "only for the current moment), so the champion here is blind to availability, "
        "while the LIVE projection already scales minutes by `chance_of_playing`. Part "
        "of this headroom is therefore already captured live. The remaining prize for "
        "predicted lineups is the part flags cannot give: **rotation** — who a manager "
        "actually picks among fit players.",
    ]
    return "\n".join(lines), headroom, per_season


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_minutes_ceiling_fast.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    champion, perfect = load_frames()
    logger.info("loaded frames: champion=%d rows, perfect=%d rows", len(champion), len(perfect))
    table = build_table(champion, perfect)
    logger.info("aligned %d gameweeks", len(table))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report, headroom, per_season = build_report(table, run_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("minutes_ceiling_fast_report_" + run_id + ".md")).write_text(
        report, encoding="utf-8")
    (RESULTS_DIR / ("minutes_ceiling_fast_" + run_id + ".json")).write_text(
        json.dumps({"headroom": headroom, "per_season": per_season}, indent=2),
        encoding="utf-8")
    print(report.encode("ascii", "replace").decode("ascii"))
    return headroom


if __name__ == "__main__":
    main()
