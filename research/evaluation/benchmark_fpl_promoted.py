"""Phase 6e Gate B: does a promoted-team Elo prior grow the FPL edge?

The shipped model cold-starts every unseen team at the league-average 1500, but
promoted teams concede ~+24% / score ~-29% and Elo (K=20) takes ~15 games to
correct - so their fixtures (favourable for the OPPONENT) are mis-ranked well into
the scored range. Gate A showed a promoted prior predicts those fixtures a little
better (best penalty=100, ~0.004 log loss, not robustly significant). This is the
test that decides: does it convert into more FPL points?

Unlike the rate-refinement nulls (decay/penalties/opponent-venue), this is a
FIXTURE-DISCRIMINATION change - getting team strength right - which is the actual
source of the FPL edge, so it earns the edge test despite a modest Gate A.

Pre-registered PRIMARY (same as Phase 5e/6b): £100m squad + captain gain over
player_ppg, 8 Understat-xG seasons / 263 GW, paired HEAD-TO-HEAD vs the CHAMPION
(recent-form minutes, no prior), which is reused unchanged so only the Elo prior
varies. Two penalties carried from Gate A (100, 150), Holm-corrected. Ships only if
a challenger beats the champion head-to-head, positive and significant on BOTH
tests after Holm, and stays non-negative without the DC-rule season.
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
from research.evaluation.benchmark_fpl_window_decay import _loso, gain_over_ppg, head_to_head

logger = logging.getLogger(__name__)

SEASONS = ALL_SEASONS
HALF_LIFE = 2.0

# champion (recent-form minutes, no prior) reused from Phase 6b Gate B
DEFAULT = ("champion (no promoted prior)", 0.0, "understat_min_hl2", "prom_def")
CHALLENGERS = [
    ("promoted prior = 100", 100.0, "understat_prom100", "prom_100"),
    ("promoted prior = 150", 150.0, "understat_prom150", "prom_150"),
]


def load_frame(penalty, cache_tag):
    return cached_predictions(seasons=SEASONS, xg_source="understat",
                              minutes_mode="recent", minutes_half_life=HALF_LIFE,
                              elo_promoted_penalty=penalty, cache_tag=cache_tag)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_promoted.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    budget, captain = PRIMARY_BUDGET, PRIMARY_CAPTAIN

    logger.info("loading CHAMPION frame (no promoted prior)")
    default_frame = load_frame(DEFAULT[1], DEFAULT[2])
    default_gain = gain_over_ppg(default_frame, DEFAULT[3], budget, captain)

    results = []
    for label, penalty, pred_tag, squad_tag in CHALLENGERS:
        logger.info("building/loading challenger frame: %s", label)
        frame = load_frame(penalty, pred_tag)
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
            "label": label, "elo_promoted_penalty": penalty,
            "head_to_head_vs_default": h2h, "head_to_head_non_dc": h2h_non_dc,
            "per_season_vs_default": per_season, "own_gain_over_ppg": own_gain,
        })

    apply_multiplicity_correction([r["head_to_head_vs_default"] for r in results])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(default_gain, results, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_promoted_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_promoted_" + run_id + ".json")).write_text(
        json.dumps({"default_gain_over_ppg": default_gain, "challengers": results}, indent=2),
        encoding="utf-8")
    print(report)
    print("\nArtifacts written to " + str(RESULTS_DIR))
    return results


def build_report(default_gain, results, run_id):
    lines = [
        "# Gate B (Phase 6e): promoted-team Elo prior on the FPL edge - " + run_id,
        "",
        "Pre-registered primary: £{0:.0f}m squad + captain, 8 Understat-xG seasons "
        "({1} GW), head-to-head vs the CHAMPION (recent-form minutes, no prior), "
        "which beats player_ppg by **{2:+.2f} pts/GW**. Only the Elo promoted prior "
        "varies.".format(PRIMARY_BUDGET, default_gain["gameweeks"], default_gain["mean_gain_per_gw"]),
        "",
        "## Each configuration's own edge over player_ppg",
        "",
        "| config | gain/GW vs ppg | t p | Wilcoxon p |",
        "|---|---|---|---|",
        "| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            "champion (no prior)", default_gain["mean_gain_per_gw"],
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
              "| season | " + " | ".join(r["label"] for r in results) + " |",
              "|---|" + "|".join("---" for _ in results) + "|"]
    for season in SEASONS:
        cells = ["{0:+.3f}".format(r["per_season_vs_default"][season]["mean_gain_per_gw"]) for r in results]
        lines.append("| {0} | {1} |".format(season, " | ".join(cells)))

    best = max(results, key=lambda r: r["head_to_head_vs_default"]["mean_gain_per_gw"])
    bh = best["head_to_head_vs_default"]
    ships = bool(bh.get("significant_corrected") and best["head_to_head_non_dc"]["mean_gain_per_gw"] >= 0)
    pooled, loso_min, loso_season = _loso(best["per_season_vs_default"])
    pos = sum(1 for s in best["per_season_vs_default"].values() if s["mean_gain_per_gw"] > 0)
    n = len(best["per_season_vs_default"])
    lines += [
        "", "## Verdict", "",
        ("**The promoted-team prior clears the bar** ({0}): it beats the champion by "
         "{1:+.3f} pts/GW head-to-head, significant on both tests after Holm, and "
         "stays positive without the DC season. The first team-model gain - adopt "
         "it and re-run the Phase 5e headline.".format(best["label"], bh["mean_gain_per_gw"])
         if ships else
         "**No challenger clears the bar.** Best arm ({0}) moves the edge by "
         "{1:+.3f} pts/GW head-to-head (t p={2:.4f}, Wilcoxon p={3:.4f}) - not "
         "significant. Getting promoted teams' strength right is more correct and "
         "improved their fixture prediction (Gate A), but by the scored range the "
         "weekly-refit model has enough of their results that the prior barely "
         "changes the squad decision. Shipped cold-start stands; recorded as a "
         "measured null.".format(best["label"], bh["mean_gain_per_gw"], bh["paired_t_p"], bh["wilcoxon_p"])),
        "",
        "**Robustness of the best arm:** positive in {0}/{1} seasons; pooled "
        "{2:+.3f}/GW, dropping its best season ({3}) leaves {4:+.3f}/GW.".format(
            pos, n, pooled, loso_season, loso_min),
        "",
        "## Caveats",
        "",
        "- Head-to-head isolates ONLY the Elo prior: identical minutes model, goal "
        "models, optimizer, budget, captain and baseline.",
        "- Penalties 100/150 carried from Gate A, not tuned on the edge; Holm across the two.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
