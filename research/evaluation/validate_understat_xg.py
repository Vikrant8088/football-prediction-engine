"""Validation gate: does Understat-sourced xG reproduce the FPL-sourced result?

Before trusting Understat's xG on the seasons where only Understat has it
(2018/19-2021/22), prove it on the seasons where BOTH sources exist (2022/23
onward). Run the identical squad backtest twice - once on FPL's own per-match xG,
once on Understat's, joined by player and date - and compare the headline the whole
project turns on: the £100m-squad, captained gain over `player_ppg`.

If the two agree, the pipeline (the player join, the date match, the substitution)
is faithful, and extending it to the earlier seasons is sound. If they diverge, the
join is lying somewhere and the extra seasons cannot be trusted - better to find
that here, on four seasons, than to discover it in a headline.

This runs only the primary cell (not the whole sweep), so it is cheap: the point is
agreement between two xG sources, not a fresh significance hunt.
"""

import logging
from pathlib import Path

from data_warehouse.utils.logging_config import configure_logging
from research.data.fpl_archive import SEASONS_WITH_XG
from research.evaluation.benchmark_fpl_optimizer import (
    PRIMARY_BUDGET,
    PRIMARY_CAPTAIN,
    compare,
    set_cache_tag,
    without_dc_season,
)
from research.evaluation.benchmark_fpl_projections import cached_predictions

logger = logging.getLogger(__name__)


def _primary(frame):
    return compare(frame, "ours", "player_ppg", PRIMARY_BUDGET, PRIMARY_CAPTAIN)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="validate_understat_xg.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    logger.info("loading FPL-xG predictions (cached)")
    fpl = cached_predictions(seasons=SEASONS_WITH_XG, xg_source="fpl")
    logger.info("building Understat-xG predictions for the same seasons")
    us = cached_predictions(seasons=SEASONS_WITH_XG, xg_source="understat")

    # Distinct cache tags: the two frames share (season, gameweek, model, budget)
    # keys, so without this the second scored frame would reuse the first's squads.
    set_cache_tag("fpl")
    fpl_primary = _primary(fpl)
    fpl_nondc = without_dc_season(fpl, PRIMARY_BUDGET, PRIMARY_CAPTAIN)
    set_cache_tag("understat")
    us_primary = _primary(us)
    us_nondc = without_dc_season(us, PRIMARY_BUDGET, PRIMARY_CAPTAIN)

    print("\n=== VALIDATION: FPL xG vs Understat xG, 2022/23-2024/25 (+2025/26)")
    print("Primary endpoint (£%.0fm squad, captain), gain over player_ppg:" % PRIMARY_BUDGET)
    print("  FPL xG      : %+.2f pts/GW  (t p=%.4f, W p=%.4f)"
          % (fpl_primary["mean_gain_per_gw"], fpl_primary["paired_t_p"],
             fpl_primary["wilcoxon_p"]))
    print("  Understat xG: %+.2f pts/GW  (t p=%.4f, W p=%.4f)"
          % (us_primary["mean_gain_per_gw"], us_primary["paired_t_p"],
             us_primary["wilcoxon_p"]))
    print("  difference  : %+.2f pts/GW" % (
        us_primary["mean_gain_per_gw"] - fpl_primary["mean_gain_per_gw"]))
    print("Non-DC seasons (fixture edge only):")
    print("  FPL xG      : %+.2f pts/GW (p=%.4f)"
          % (fpl_nondc["mean_gain_per_gw"], fpl_nondc["paired_t_p"]))
    print("  Understat xG: %+.2f pts/GW (p=%.4f)"
          % (us_nondc["mean_gain_per_gw"], us_nondc["paired_t_p"]))

    gap = abs(us_primary["mean_gain_per_gw"] - fpl_primary["mean_gain_per_gw"])
    same_sign = (us_primary["mean_gain_per_gw"] > 0) == (fpl_primary["mean_gain_per_gw"] > 0)
    # Agreement bar: same direction, and within 1.0 pt/GW. xG providers differ a
    # little by construction, so exact equality is not expected - direction and
    # magnitude are.
    verdict = "PASS" if same_sign and gap <= 1.0 else "FAIL"
    print("\nVERDICT: %s  (same direction=%s, gap=%.2f <= 1.0)" % (verdict, same_sign, gap))
    return verdict


if __name__ == "__main__":
    main()
