"""Phase 6a Stage 2: does the window/decay change grow the FPL pts/GW edge?

Stage 1 (benchmark_window_decay) screened the two goal models on the scoreline-
grid channel and found a real, directionally-coherent signal: the shipped Dixon-
Coles decay (xi=0.0065) is too aggressive - it discards almost everything older
than a season - and a gentler xi ~= 0.001 predicts clean sheets AND 1X2 better;
the decay-less Poisson likewise prefers a bounded ~5-season window over an
expanding one. Two caveats were logged before this stage: the DC clean-sheet gain
passed the t-test but FAILED Wilcoxon (concentrated, not broad), and the FPL grid
rescales the goal-model grid to Elo-led marginals, which dampens the effect. So
the honest prior here is "small, possibly null" - which is exactly why it must be
measured on the endpoint that decides, not asserted.

This is the pre-registered PRIMARY endpoint from the window/decay pre-registration:
the GBP100m squad + captain gain over player_ppg, on the same 8 Understat-xG
seasons as the Phase 5e headline. Two challenger team models are carried from the
screen, each vs the shipped default:

  DEFAULT   dc_xi=0.0065 (library default), expanding window   <- ships today
  C1 decay  dc_xi=0.001,  expanding window
  C2 win+dec dc_xi=0.001,  5-season rolling window

The decisive test is the paired HEAD-TO-HEAD: per gameweek, does the challenger's
optimal squad's ACTUAL points exceed the default's? Because both configs are
scored against the identical player_ppg baseline, (gain_challenger - gain_default)
equals (ours_challenger - ours_default) head-to-head - so this is exactly "beats
the default configuration on the primary gain." A challenger REPLACES the default
only if it wins here, positive and significant on BOTH paired t and Wilcoxon after
Holm correction across the two challengers, and non-negative once the 2025/26
defensive-contribution season is removed. Otherwise the default stands and the
lever is recorded as a null.

Every projection frame gets its own squad-cache tag: the in-memory squad cache is
keyed by (season, gameweek, model, budget, tag) and NOT by which frame produced
the projection, so without distinct tags the second config would silently reuse
the first's squads. That exact collision produced a spuriously perfect validation
result once already; it is not repeated here.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from research.data.fpl_archive import ALL_SEASONS
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_optimizer import (
    DC_SEASON,
    PRIMARY_BUDGET,
    PRIMARY_CAPTAIN,
    _squad_points,
    apply_multiplicity_correction,
    compare,
    set_cache_tag,
)
from research.evaluation.benchmark_fpl_projections import cached_predictions

logger = logging.getLogger(__name__)

XG_SOURCE = "understat"          # the 8-season Phase 5e basis
SEASONS = ALL_SEASONS

# (label, dc_xi, max_train_seasons, predictions-cache-tag, squad-cache-tag)
DEFAULT = ("default (xi=0.0065, expanding)", None, None, "understat", "wd_def")

# EXPLORATORY follow-up (NOT the original pre-registration): the pre-registered
# C1/C2 arms (gentler decay) were a null - worse, in fact - but revealed the FPL
# edge RISES with xi over [0.001, 0.0065] (+4.17 -> +5.02/GW). The edge lives in
# fixture reactivity, which a MORE aggressive decay serves, opposite to what the
# clean-sheet calibration metric suggested. So push decay harder and see if the
# edge keeps climbing. A win here is IN-SAMPLE selected and would need its own
# pre-registered confirmation before shipping.
CHALLENGERS = [
    ("E1 aggressive (xi=0.012, expanding)", 0.012, None, "understat_e1_xi012", "wd_e1"),
    ("E2 very aggressive (xi=0.02, expanding)", 0.02, None, "understat_e2_xi020", "wd_e2"),
]


def load_frame(dc_xi, max_train_seasons, cache_tag):
    return cached_predictions(seasons=SEASONS, xg_source=XG_SOURCE, dc_xi=dc_xi,
                              max_train_seasons=max_train_seasons, cache_tag=cache_tag)


def _paired(a: pd.Series, b: pd.Series) -> dict:
    """Paired stats of squad-points series a (challenger) minus b (default),
    aligned on their common (season, gameweek) index."""
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    diff = a - b
    t_p = float(stats.ttest_rel(a, b)[1]) if len(diff) > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1])
    except ValueError:
        w_p = float("nan")
    return {
        "gameweeks": int(len(diff)),
        "challenger_points_per_gw": float(a.mean()),
        "default_points_per_gw": float(b.mean()),
        "mean_gain_per_gw": float(diff.mean()),
        "median_gain_per_gw": float(np.median(diff)) if len(diff) else float("nan"),
        "gameweeks_won": int((diff > 0).sum()),
        "gameweeks_lost": int((diff < 0).sum()),
        "season_gain_per_38_gw": float(diff.mean() * 38),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "significant": bool(diff.mean() > 0 and t_p < 0.05 and w_p < 0.05),
    }


def head_to_head(frame_ch, tag_ch, frame_def, tag_def, budget, captain) -> dict:
    """Per-gameweek: challenger's optimal-squad actual points minus the default's.
    Distinct squad-cache tags per frame (see module docstring)."""
    set_cache_tag(tag_ch)
    a = _squad_points(frame_ch, "ours", budget, captain)
    set_cache_tag(tag_def)
    b = _squad_points(frame_def, "ours", budget, captain)
    return _paired(a, b)


def gain_over_ppg(frame, squad_tag, budget, captain) -> dict:
    """Each config's own gain over player_ppg - the Phase 5e framing, so the two
    configs' edges are directly comparable to the +4.99/GW headline."""
    set_cache_tag(squad_tag)
    return compare(frame, "ours", "player_ppg", budget, captain)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_window_decay.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    budget, captain = PRIMARY_BUDGET, PRIMARY_CAPTAIN

    logger.info("loading DEFAULT frame (cached Phase 5e)")
    default_frame = load_frame(*DEFAULT[1:4])

    logger.info("DEFAULT gain over player_ppg (primary framing)")
    default_gain = gain_over_ppg(default_frame, DEFAULT[4], budget, captain)

    results = []
    for label, dc_xi, window, pred_tag, squad_tag in CHALLENGERS:
        logger.info("building/loading challenger frame: %s", label)
        frame = load_frame(dc_xi, window, pred_tag)

        h2h = head_to_head(frame, squad_tag, default_frame, DEFAULT[4], budget, captain)
        h2h_non_dc = head_to_head(
            frame[frame["season"] != DC_SEASON], squad_tag,
            default_frame[default_frame["season"] != DC_SEASON], DEFAULT[4],
            budget, captain)
        per_season = {}
        for season in SEASONS:
            per_season[season] = head_to_head(
                frame[frame["season"] == season], squad_tag,
                default_frame[default_frame["season"] == season], DEFAULT[4],
                budget, captain)
        own_gain = gain_over_ppg(frame, squad_tag, budget, captain)

        results.append({
            "label": label, "dc_xi": dc_xi, "max_train_seasons": window,
            "head_to_head_vs_default": h2h,
            "head_to_head_non_dc": h2h_non_dc,
            "per_season_vs_default": per_season,
            "own_gain_over_ppg": own_gain,
        })

    # Holm correction across the two challengers, on the head-to-head vs default.
    h2h_cells = [r["head_to_head_vs_default"] for r in results]
    apply_multiplicity_correction(h2h_cells)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(default_gain, results, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("fpl_window_decay_report_" + run_id + ".md")).write_text(
        report, encoding="utf-8")
    (RESULTS_DIR / ("fpl_window_decay_" + run_id + ".json")).write_text(
        json.dumps({"default_gain_over_ppg": default_gain, "challengers": results},
                   indent=2), encoding="utf-8")
    print(report)
    print("\nArtifacts written to " + str(RESULTS_DIR))
    return results


def build_report(default_gain, results, run_id) -> str:
    lines = [
        "# Phase 6a Stage 2: window/decay on the FPL pts/GW edge - " + run_id,
        "",
        "Pre-registered primary endpoint: the GBP{0:.0f}m squad + captain, "
        "8 Understat-xG seasons ({1} GW). The shipped team model (Dixon-Coles "
        "xi=0.0065, expanding window) already beats player_ppg by "
        "**{2:+.2f} pts/GW** here (t p={3:.4f}, Wilcoxon p={4:.4f}) - this is the "
        "Phase 5e headline, reproduced as the baseline to improve on.".format(
            PRIMARY_BUDGET, default_gain["gameweeks"],
            default_gain["mean_gain_per_gw"], default_gain["paired_t_p"],
            default_gain["wilcoxon_p"]),
        "",
        "The question: does changing the Dixon-Coles decay and/or the training "
        "window grow that edge? The decisive column is the paired **head-to-head** - challenger "
        "squad points minus default squad points, gameweek by gameweek. A "
        "challenger ships only if it is positive and significant on BOTH tests "
        "after Holm correction across the two challengers, and does not turn "
        "negative once the 2025/26 defensive-contribution season is removed.",
        "",
        "## Each configuration's own edge over player_ppg",
        "",
        "| config | gain/GW vs ppg | t p | Wilcoxon p |",
        "|---|---|---|---|",
        "| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            "default (xi=0.0065, expanding)", default_gain["mean_gain_per_gw"],
            default_gain["paired_t_p"], default_gain["wilcoxon_p"]),
    ]
    for r in results:
        g = r["own_gain_over_ppg"]
        lines.append("| {0} | **{1:+.2f}** | {2:.4f} | {3:.4f} |".format(
            r["label"], g["mean_gain_per_gw"], g["paired_t_p"], g["wilcoxon_p"]))

    lines += [
        "",
        "## The decisive test: challenger vs default, head-to-head",
        "",
        "| challenger | h2h gain/GW | GWs won/lost | t p | Wilcoxon p | "
        "Holm survives? | non-DC gain/GW | ships? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        h = r["head_to_head_vs_default"]
        nd = r["head_to_head_non_dc"]
        ships = bool(h.get("significant_corrected") and nd["mean_gain_per_gw"] >= 0)
        lines.append(
            "| {0} | **{1:+.3f}** | {2}/{3} | {4:.4f} | {5:.4f} | {6} | "
            "{7:+.3f} | {8} |".format(
                r["label"], h["mean_gain_per_gw"], h["gameweeks_won"],
                h["gameweeks_lost"], h["paired_t_p"], h["wilcoxon_p"],
                "yes" if h.get("significant_corrected") else "no",
                nd["mean_gain_per_gw"], "**YES**" if ships else "no"))

    lines += ["", "## Per-season replication (head-to-head vs default)", "",
              "| season | " + " | ".join(r["label"].split(" (")[0] for r in results) + " |",
              "|---|" + "|".join("---" for _ in results) + "|"]
    for season in SEASONS:
        cells = []
        for r in results:
            s = r["per_season_vs_default"][season]
            cells.append("{0:+.3f}".format(s["mean_gain_per_gw"]))
        lines.append("| {0} | {1} |".format(season, " | ".join(cells)))

    best = max(results, key=lambda r: r["head_to_head_vs_default"]["mean_gain_per_gw"])
    bh = best["head_to_head_vs_default"]
    ships = bool(bh.get("significant_corrected")
                 and best["head_to_head_non_dc"]["mean_gain_per_gw"] >= 0)
    pooled, loso_min, loso_season = _loso(best["per_season_vs_default"])
    pos_seasons = sum(1 for s in best["per_season_vs_default"].values()
                      if s["mean_gain_per_gw"] > 0)
    n_seasons = len(best["per_season_vs_default"])

    lines += [
        "",
        "## Verdict",
        "",
        ("**A challenger clears the pre-registered bar** ({0}): it beats the shipped "
         "team model head-to-head by {1:+.3f} pts/GW, significant on both tests after "
         "Holm correction, and stays positive without the DC-rule season. Adopt it "
         "and re-run the Phase 5e headline to confirm the new edge.".format(
             best["label"], bh["mean_gain_per_gw"])
         if ships else
         "**No challenger clears the bar.** The best arm ({0}) beats the shipped "
         "default by {1:+.3f} pts/GW head-to-head, but that is NOT significant "
         "(t p={2:.4f}, Wilcoxon p={3:.4f}) and does not survive Holm correction. "
         "The shipped xi=0.0065 / expanding-window default stands; on the FPL "
         "endpoint the window/decay lever does not clear the project's bar.".format(
             best["label"], bh["mean_gain_per_gw"], bh["paired_t_p"],
             bh["wilcoxon_p"])),
        "",
        "**Robustness of the best arm ({0}):** positive in {1}/{2} seasons; "
        "gameweek-weighted pooled gain {3:+.3f}/GW, but dropping its single most "
        "favourable season ({4}) leaves only {5:+.3f}/GW. {6}".format(
            best["label"], pos_seasons, n_seasons, pooled, loso_season, loso_min,
            "A large share of the gain rides on one season - the Phase 5b failure "
            "mode - so even the point estimate is fragile."
            if pooled > 1e-9 and loso_min < 0.5 * pooled else
            "The gain does not collapse to one season."),
        "",
        "## Caveats",
        "",
        "- Head-to-head isolates ONLY the team-model change: both arms share the "
        "identical player rates, optimizer, budget, captain rule and player_ppg "
        "baseline, so any difference is the decay/window and nothing else.",
        "- Squad-cache tags are distinct per frame, so no arm reuses another's "
        "squads (the collision that once faked a perfect result).",
        "- Any positive point estimate here was found by searching xi in the "
        "promising direction, so it is IN-SAMPLE selected; shipping it would "
        "require a leakage-safe nested-tuning confirmation, not this number.",
    ]
    return "\n".join(lines)


def _loso(per_season: dict):
    """Gameweek-weighted pooled head-to-head gain, and the minimum after dropping
    any single season - the leave-one-season-out fragility check."""
    items = [(name, s["mean_gain_per_gw"], s["gameweeks"])
             for name, s in per_season.items() if s["gameweeks"] > 0]
    total_gw = sum(gw for _, _, gw in items)
    if total_gw == 0:
        return 0.0, 0.0, "n/a"
    pooled = sum(m * gw for _, m, gw in items) / total_gw
    worst_season, loso_min = "n/a", pooled
    for drop, _, _ in items:
        kept = [(m, gw) for name, m, gw in items if name != drop]
        gw_kept = sum(gw for _, gw in kept)
        if gw_kept == 0:
            continue
        val = sum(m * gw for m, gw in kept) / gw_kept
        if val < loso_min:
            loso_min, worst_season = val, drop
    return pooled, loso_min, worst_season


if __name__ == "__main__":
    main()
