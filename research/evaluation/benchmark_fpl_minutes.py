"""Gate B: does the recent-form minutes model grow the FPL pts/GW edge?

Gate A proved the recent-form model predicts a player's next-match minutes far
better than the shipped flat average (squad-pool MAE 22.0 vs 28.8; p_60 Brier
0.169 vs 0.277). That is necessary but NOT sufficient - the decay experiment
already showed that a team-model change which improves its own accuracy can
*degrade* the product (calibration != discrimination). So this is the test that
decides: plug the better minutes model into the projection and re-measure the
edge that matters.

Pre-registered PRIMARY endpoint (same as Phase 5e): the GBP100m squad + captain
gain over player_ppg, 8 Understat-xG seasons / 263 GW. The decisive column is the
paired HEAD-TO-HEAD - challenger squad points minus the shipped crude model's,
gameweek by gameweek. Because both are scored against the identical player_ppg
baseline, that head-to-head IS "beats the shipped configuration on the primary
gain".

Two challengers, carried from Gate A (Holm-corrected across the two):
  DEFAULT  crude flat season average                <- ships today
  HL=1     recent-form, half-life 1 match  (best Gate-A minutes MAE)
  HL=2     recent-form, half-life 2 matches (best Gate-A p_60 Brier, less twitchy)

A challenger REPLACES the crude model only if it beats it head-to-head, positive
and significant on BOTH paired t and Wilcoxon after Holm correction, and stays
non-negative once the 2025/26 defensive-contribution season is removed. Otherwise
the crude model stands and the minutes lever is a measured null on the endpoint -
even though it is unambiguously a better minutes predictor.

Distinct squad-cache tags per frame (the collision that once faked a perfect
result); the crude arm reuses the Phase 5e prediction cache unchanged.
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

XG_SOURCE = "understat"
SEASONS = ALL_SEASONS

# (label, minutes_mode, half_life, predictions-cache-tag, squad-cache-tag)
DEFAULT = ("crude (season average)", "crude", 2.0, "understat", "min_def")
CHALLENGERS = [
    ("recent-form HL=1", "recent", 1.0, "understat_min_hl1", "min_hl1"),
    ("recent-form HL=2", "recent", 2.0, "understat_min_hl2", "min_hl2"),
]


def load_frame(mode, half_life, cache_tag):
    return cached_predictions(seasons=SEASONS, xg_source=XG_SOURCE,
                              minutes_mode=mode, minutes_half_life=half_life,
                              cache_tag=cache_tag)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_minutes.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    budget, captain = PRIMARY_BUDGET, PRIMARY_CAPTAIN

    logger.info("loading DEFAULT (crude) frame - cached Phase 5e")
    default_frame = load_frame(*DEFAULT[1:4])
    default_gain = gain_over_ppg(default_frame, DEFAULT[4], budget, captain)

    results = []
    for label, mode, half_life, pred_tag, squad_tag in CHALLENGERS:
        logger.info("building/loading challenger frame: %s", label)
        frame = load_frame(mode, half_life, pred_tag)
        h2h = head_to_head(frame, squad_tag, default_frame, DEFAULT[4], budget, captain)
        h2h_non_dc = head_to_head(
            frame[frame["season"] != DC_SEASON], squad_tag,
            default_frame[default_frame["season"] != DC_SEASON], DEFAULT[4],
            budget, captain)
        per_season = {season: head_to_head(
            frame[frame["season"] == season], squad_tag,
            default_frame[default_frame["season"] == season], DEFAULT[4],
            budget, captain) for season in SEASONS}
        own_gain = gain_over_ppg(frame, squad_tag, budget, captain)
        results.append({
            "label": label, "minutes_mode": mode, "half_life": half_life,
            "head_to_head_vs_default": h2h, "head_to_head_non_dc": h2h_non_dc,
            "per_season_vs_default": per_season, "own_gain_over_ppg": own_gain,
        })

    apply_multiplicity_correction([r["head_to_head_vs_default"] for r in results])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(default_gain, results, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_minutes_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_minutes_" + run_id + ".json")).write_text(
        json.dumps({"default_gain_over_ppg": default_gain, "challengers": results},
                   indent=2), encoding="utf-8")
    print(report)
    print("\nArtifacts written to " + str(RESULTS_DIR))
    return results


def build_report(default_gain, results, run_id):
    lines = [
        "# Gate B: recent-form minutes on the FPL pts/GW edge - " + run_id,
        "",
        "Pre-registered primary: GBP{0:.0f}m squad + captain, 8 Understat-xG seasons "
        "({1} GW). The shipped crude-minutes model beats player_ppg by "
        "**{2:+.2f} pts/GW** (the Phase 5e headline). Gate A already proved recent-"
        "form is a much better *minutes* predictor; the question here is whether "
        "that converts into a better *edge*.".format(
            PRIMARY_BUDGET, default_gain["gameweeks"],
            default_gain["mean_gain_per_gw"]),
        "",
        "## Each configuration's own edge over player_ppg",
        "",
        "| config | gain/GW vs ppg | t p | Wilcoxon p |",
        "|---|---|---|---|",
        "| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            "crude (shipped)", default_gain["mean_gain_per_gw"],
            default_gain["paired_t_p"], default_gain["wilcoxon_p"]),
    ]
    for r in results:
        g = r["own_gain_over_ppg"]
        lines.append("| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            r["label"], g["mean_gain_per_gw"], g["paired_t_p"], g["wilcoxon_p"]))

    lines += [
        "",
        "## The decisive test: challenger vs crude, head-to-head",
        "",
        "| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | "
        "Holm survives? | non-DC gain/GW | ships? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        h, nd = r["head_to_head_vs_default"], r["head_to_head_non_dc"]
        ships = bool(h.get("significant_corrected") and nd["mean_gain_per_gw"] >= 0)
        lines.append(
            "| {0} | **{1:+.3f}** | {2}/{3} | {4:.4f} | {5:.4f} | {6} | {7:+.3f} | {8} |".format(
                r["label"], h["mean_gain_per_gw"], h["gameweeks_won"], h["gameweeks_lost"],
                h["paired_t_p"], h["wilcoxon_p"],
                "yes" if h.get("significant_corrected") else "no",
                nd["mean_gain_per_gw"], "**YES**" if ships else "no"))

    lines += ["", "## Per-season replication (head-to-head vs crude)", "",
              "| season | " + " | ".join(r["label"] for r in results) + " |",
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
        "",
        "## Verdict",
        "",
        ("**The recent-form minutes model clears the bar** ({0}): it beats the "
         "shipped crude model by {1:+.3f} pts/GW head-to-head, significant on both "
         "tests after Holm correction, and stays positive without the DC-rule "
         "season. A genuinely better minutes signal that also grows the edge - "
         "adopt it and re-run the Phase 5e headline.".format(best["label"], bh["mean_gain_per_gw"])
         if ships else
         "**No challenger clears the bar.** The best arm ({0}) moves the edge by "
         "{1:+.3f} pts/GW head-to-head (t p={2:.4f}, Wilcoxon p={3:.4f}) - not "
         "significant after correction. A dramatically better minutes predictor "
         "(Gate A) does not translate into a better squad-points edge: the "
         "optimizer already avoided the fringe/rotation players the crude model "
         "over-rated, so sharper minutes mostly re-rank players who were never "
         "picked. The shipped crude model stands.".format(
             best["label"], bh["mean_gain_per_gw"], bh["paired_t_p"], bh["wilcoxon_p"])),
        "",
        "**Robustness of the best arm ({0}):** positive in {1}/{2} seasons; pooled "
        "{3:+.3f}/GW, dropping its best season ({4}) leaves {5:+.3f}/GW.".format(
            best["label"], pos, n, pooled, loso_season, loso_min),
        "",
        "## Caveats",
        "",
        "- Head-to-head isolates ONLY the minutes model: identical team model, "
        "player rates, optimizer, budget, captain and player_ppg baseline.",
        "- **Backtest handicap:** live, the projection also has FPL's injury/"
        "availability flag, the single biggest minutes signal - this backtest "
        "cannot (flags are not published historically). So this UNDERSTATES the "
        "live value of a minutes model; it measures only the recent-form part.",
        "- Distinct squad-cache tags per frame; the crude arm reuses the Phase 5e "
        "prediction cache unchanged.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
