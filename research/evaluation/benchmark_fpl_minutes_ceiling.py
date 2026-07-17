"""The minutes ceiling: how much edge could PERFECT lineup knowledge ever add?

Live predicted lineups are the research's #1 remaining FPL signal, but they cannot
be backtested (nobody archived predicted XIs). Before investing in a scraper for
imperfect, unvalidatable data, this bounds the whole opportunity: give the minutes
model PERFECT foreknowledge of exactly how long every player actually played this
gameweek, and measure how much the edge grows over the shipped recent-form model.

This deliberately LEAKS the gameweek's actual minutes - it is an upper bound, not a
shippable model. Any real lineup signal (predicted XIs included) is strictly worse
than perfect, so:
  - if perfect minutes barely beats recent-form -> live lineups cannot help much,
    don't build the scraper;
  - if perfect minutes adds a lot -> there is real headroom worth chasing.

Head-to-head vs the recent-form champion (reused unchanged), 8 Understat-xG seasons.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from data_warehouse.utils.logging_config import configure_logging
from research.data.fpl_archive import ALL_SEASONS
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_optimizer import (
    DC_SEASON,
    PRIMARY_BUDGET,
    PRIMARY_CAPTAIN,
)
from research.evaluation.benchmark_fpl_projections import cached_predictions
from research.evaluation.benchmark_fpl_window_decay import _loso, gain_over_ppg, head_to_head

logger = logging.getLogger(__name__)

SEASONS = ALL_SEASONS
# champion (recent-form minutes) reused from Phase 6b Gate B
DEFAULT = ("recent-form minutes (shipped)", "recent", "understat_min_hl2", "ceil_def")
PERFECT = ("PERFECT minutes (ceiling, leaks actuals)", "perfect", "understat_perfect", "ceil_perfect")


def load_frame(mode, cache_tag):
    return cached_predictions(seasons=SEASONS, xg_source="understat",
                              minutes_mode=mode, minutes_half_life=2.0, cache_tag=cache_tag)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_minutes_ceiling.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    budget, captain = PRIMARY_BUDGET, PRIMARY_CAPTAIN

    logger.info("loading recent-form champion frame")
    champ = load_frame(DEFAULT[1], DEFAULT[2])
    champ_gain = gain_over_ppg(champ, DEFAULT[3], budget, captain)

    logger.info("building PERFECT-minutes frame (ceiling)")
    perfect = load_frame(PERFECT[1], PERFECT[2])
    perfect_gain = gain_over_ppg(perfect, PERFECT[3], budget, captain)

    h2h = head_to_head(perfect, PERFECT[3], champ, DEFAULT[3], budget, captain)
    h2h_non_dc = head_to_head(
        perfect[perfect["season"] != DC_SEASON], PERFECT[3],
        champ[champ["season"] != DC_SEASON], DEFAULT[3], budget, captain)
    per_season = {s: head_to_head(
        perfect[perfect["season"] == s], PERFECT[3],
        champ[champ["season"] == s], DEFAULT[3], budget, captain) for s in SEASONS}
    pooled, loso_min, loso_season = _loso(per_season)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "# The minutes ceiling: perfect vs recent-form - " + run_id,
        "",
        "Perfect minutes LEAKS each gameweek's actual minutes - an upper bound on any "
        "lineup/minutes signal, not a shippable model. £{0:.0f}m squad + captain, "
        "{1} GW.".format(PRIMARY_BUDGET, champ_gain["gameweeks"]),
        "",
        "| minutes model | edge vs player_ppg |",
        "|---|---|",
        "| recent-form (shipped champion) | **{0:+.2f}/GW** |".format(champ_gain["mean_gain_per_gw"]),
        "| PERFECT (ceiling) | **{0:+.2f}/GW** |".format(perfect_gain["mean_gain_per_gw"]),
        "",
        "## Perfect vs recent-form, head-to-head (the headroom)",
        "",
        "- **Ceiling headroom: {0:+.3f} pts/GW** (t p={1:.4f}, Wilcoxon p={2:.4f}), "
        "won {3}/{4} GWs; non-DC {5:+.3f}/GW.".format(
            h2h["mean_gain_per_gw"], h2h["paired_t_p"], h2h["wilcoxon_p"],
            h2h["gameweeks_won"], h2h["gameweeks"], h2h_non_dc["mean_gain_per_gw"]),
        "- Positive in {0}/8 seasons; pooled {1:+.3f}/GW, drop best season ({2}) -> "
        "{3:+.3f}/GW.".format(
            sum(1 for v in per_season.values() if v["mean_gain_per_gw"] > 0),
            pooled, loso_season, loso_min),
        "",
        "## Per-season headroom",
        "",
        "| season | perfect - recent-form (pts/GW) |",
        "|---|---|",
    ]
    for s in SEASONS:
        lines.append("| {0} | {1:+.3f} |".format(s, per_season[s]["mean_gain_per_gw"]))
    lines += [
        "",
        "## Reading it",
        "",
        "This headroom ({0:+.2f}/GW) is the MAXIMUM any minutes/lineup data could add "
        "on top of the recent-form model. Live predicted lineups are imperfect, so "
        "they would capture only a FRACTION of it. If the headroom is small, live "
        "lineups are not worth a non-backtestable scraper; if large, there is real "
        "room to chase.".format(h2h["mean_gain_per_gw"]),
    ]
    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("minutes_ceiling_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("minutes_ceiling_" + run_id + ".json")).write_text(
        json.dumps({"champion_gain": champ_gain, "perfect_gain": perfect_gain,
                    "head_to_head": h2h, "non_dc": h2h_non_dc, "per_season": per_season}, indent=2),
        encoding="utf-8")
    print(report)
    return h2h


if __name__ == "__main__":
    main()
