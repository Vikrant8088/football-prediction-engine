"""Phase 6c: does modelling penalties separately grow the FPL edge?

A player's xG rate already INCLUDES the penalties he takes (a penalty is ~0.76
xG). The shipped model then scales that whole rate - penalties and open play alike
- by the open-play fixture multiplier, which over-inflates a penalty taker's
projection in his easy fixtures (as if the team wins proportionally more penalties
the more open-play goals it is expected to score). Understat's per-match npxG
(non-penalty xG) lets us split them: penalty xG = xG - npxG, leakage-safe and
per-match, and a player only accrues it when he actually takes the kick - so it
doubles as the taker identifier.

The change: open-play xG keeps the fixture multiplier; penalty xG uses its own
(penalty_multiplier = 1.0 here - a penalty's value hangs on who takes it and
whether one is awarded, not on the open-play match-up). This is the single,
pre-specified hypothesis; the multiplier is not tuned on the edge.

Pre-registered PRIMARY endpoint (same as Phase 5e/6b): the £100m squad + captain
gain over player_ppg, 8 Understat-xG seasons / 263 GW, paired HEAD-TO-HEAD vs the
current CHAMPION (recent-form minutes, no penalty split). The champion frame is
reused unchanged as the baseline, so the only thing that varies is the penalty
split. Ships only if the challenger beats the champion head-to-head, positive and
significant on BOTH tests, and stays non-negative without the DC-rule season.

Honest prior: the correction is small (penalty xG is a fraction of a taker's rate)
and the deep-research pass found no verified penalty effect - a null here would be
unsurprising and is a first-class outcome.
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
    apply_multiplicity_correction,
)
from research.evaluation.benchmark_fpl_projections import cached_predictions
from research.evaluation.benchmark_fpl_window_decay import (
    _loso,
    gain_over_ppg,
    head_to_head,
)

logger = logging.getLogger(__name__)

SEASONS = ALL_SEASONS
HALF_LIFE = 2.0        # the shipped champion minutes model

# The champion (recent-form minutes, no penalty split) is reused from Phase 6b's
# Gate B, so the baseline is exactly what ships and only the penalty split varies.
DEFAULT = ("champion (recent-form minutes, penalties lumped)", False, "understat_min_hl2", "pen_def")
CHALLENGERS = [
    ("penalty split (open-play scaled, penalties flat)", True, "understat_pen_split", "pen_split"),
]


def load_frame(penalty_split, cache_tag):
    return cached_predictions(seasons=SEASONS, xg_source="understat",
                              minutes_mode="recent", minutes_half_life=HALF_LIFE,
                              penalty_split=penalty_split, penalty_multiplier=1.0,
                              cache_tag=cache_tag)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_penalties.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    budget, captain = PRIMARY_BUDGET, PRIMARY_CAPTAIN

    logger.info("loading CHAMPION frame (recent-form minutes, no penalty split)")
    default_frame = load_frame(DEFAULT[1], DEFAULT[2])
    default_gain = gain_over_ppg(default_frame, DEFAULT[3], budget, captain)

    results = []
    for label, penalty_split, pred_tag, squad_tag in CHALLENGERS:
        logger.info("building/loading challenger frame: %s", label)
        frame = load_frame(penalty_split, pred_tag)
        h2h = head_to_head(frame, squad_tag, default_frame, DEFAULT[3], budget, captain)
        h2h_non_dc = head_to_head(
            frame[frame["season"] != DC_SEASON], squad_tag,
            default_frame[default_frame["season"] != DC_SEASON], DEFAULT[3], budget, captain)
        per_season = {season: head_to_head(
            frame[frame["season"] == season], squad_tag,
            default_frame[default_frame["season"] == season], DEFAULT[3], budget, captain)
            for season in SEASONS}
        own_gain = gain_over_ppg(frame, squad_tag, budget, captain)
        results.append({
            "label": label, "penalty_split": penalty_split,
            "head_to_head_vs_default": h2h, "head_to_head_non_dc": h2h_non_dc,
            "per_season_vs_default": per_season, "own_gain_over_ppg": own_gain,
        })

    apply_multiplicity_correction([r["head_to_head_vs_default"] for r in results])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(default_gain, results, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_penalties_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_penalties_" + run_id + ".json")).write_text(
        json.dumps({"default_gain_over_ppg": default_gain, "challengers": results}, indent=2),
        encoding="utf-8")
    print(report)
    print("\nArtifacts written to " + str(RESULTS_DIR))
    return results


def build_report(default_gain, results, run_id):
    lines = [
        "# Gate B (Phase 6c): penalty split on the FPL pts/GW edge - " + run_id,
        "",
        "Pre-registered primary: £{0:.0f}m squad + captain, 8 Understat-xG seasons "
        "({1} GW), head-to-head vs the CHAMPION (recent-form minutes, penalties "
        "lumped), which already beats player_ppg by **{2:+.2f} pts/GW**. Only the "
        "penalty split varies.".format(PRIMARY_BUDGET, default_gain["gameweeks"],
                                       default_gain["mean_gain_per_gw"]),
        "",
        "## Each configuration's own edge over player_ppg",
        "",
        "| config | gain/GW vs ppg | t p | Wilcoxon p |",
        "|---|---|---|---|",
        "| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            "champion (penalties lumped)", default_gain["mean_gain_per_gw"],
            default_gain["paired_t_p"], default_gain["wilcoxon_p"]),
    ]
    for r in results:
        g = r["own_gain_over_ppg"]
        lines.append("| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            r["label"], g["mean_gain_per_gw"], g["paired_t_p"], g["wilcoxon_p"]))

    lines += ["", "## The decisive test: challenger vs champion, head-to-head", "",
              "| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | "
              "Holm survives? | non-DC gain/GW | ships? |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        h, nd = r["head_to_head_vs_default"], r["head_to_head_non_dc"]
        ships = bool(h.get("significant_corrected") and nd["mean_gain_per_gw"] >= 0)
        lines.append("| {0} | **{1:+.3f}** | {2}/{3} | {4:.4f} | {5:.4f} | {6} | {7:+.3f} | {8} |".format(
            r["label"], h["mean_gain_per_gw"], h["gameweeks_won"], h["gameweeks_lost"],
            h["paired_t_p"], h["wilcoxon_p"],
            "yes" if h.get("significant_corrected") else "no",
            nd["mean_gain_per_gw"], "**YES**" if ships else "no"))

    lines += ["", "## Per-season replication (head-to-head vs champion)", "",
              "| season | " + " | ".join("h2h gain/GW" for _ in results) + " |",
              "|---|" + "|".join("---" for _ in results) + "|"]
    for season in SEASONS:
        cells = ["{0:+.3f}".format(r["per_season_vs_default"][season]["mean_gain_per_gw"])
                 for r in results]
        lines.append("| {0} | {1} |".format(season, " | ".join(cells)))

    best = max(results, key=lambda r: r["head_to_head_vs_default"]["mean_gain_per_gw"])
    bh = best["head_to_head_vs_default"]
    ships = bool(bh.get("significant_corrected")
                 and best["head_to_head_non_dc"]["mean_gain_per_gw"] >= 0)
    pooled, loso_min, loso_season = _loso(best["per_season_vs_default"])
    pos = sum(1 for s in best["per_season_vs_default"].values() if s["mean_gain_per_gw"] > 0)
    n = len(best["per_season_vs_default"])
    lines += [
        "", "## Verdict", "",
        ("**The penalty split clears the bar**: it beats the champion by "
         "{0:+.3f} pts/GW head-to-head, significant on both tests, and stays "
         "positive without the DC season. Adopt it.".format(bh["mean_gain_per_gw"])
         if ships else
         "**The penalty split does not clear the bar.** It moves the edge by "
         "{0:+.3f} pts/GW head-to-head (t p={1:.4f}, Wilcoxon p={2:.4f}) - not "
         "significant. Splitting penalties from open-play xG is more principled "
         "and marginally changes premium takers' projections, but the correction "
         "is too small to move the squad decision measurably. The lumped model "
         "stands; recorded as a measured null (consistent with the research finding "
         "no verified penalty effect and with xG already capturing takers).".format(
             bh["mean_gain_per_gw"], bh["paired_t_p"], bh["wilcoxon_p"])),
        "",
        "**Robustness of the best arm:** positive in {0}/{1} seasons; pooled "
        "{2:+.3f}/GW, dropping its best season ({3}) leaves {4:+.3f}/GW.".format(
            pos, n, pooled, loso_season, loso_min),
        "",
        "## Caveats",
        "",
        "- Head-to-head isolates ONLY the penalty split: identical minutes model, "
        "team model, optimizer, budget, captain and baseline.",
        "- penalty_multiplier=1.0 is pre-specified (penalties fixture-independent), "
        "not tuned on the edge.",
        "- Backtest handicap unchanged: no live penalty-taker feed; the split is "
        "inferred from each player's own realised penalty xG.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
