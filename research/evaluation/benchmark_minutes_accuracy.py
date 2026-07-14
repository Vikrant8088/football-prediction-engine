"""Gate A: does a recent-form minutes model predict next-match minutes better
than the shipped flat season average?

This is the cheap gate before touching the projection. It uses ONLY the per-
gameweek minutes history (no team model, no xG), walk-forward WITHIN each season
(minutes history resets per season, exactly as the FPL backtest does), and asks a
direct question: at gameweek k, using only minutes from gameweeks 1..k-1, whose
prediction of the player's ACTUAL minutes in gameweek k is closer -

  crude    total prior minutes / prior matches           (the shipped model)
  recent   recency-weighted, various half-lives           (the candidate)

Metrics (lower is better):
  minutes MAE     |predicted expected_minutes - actual minutes|
  p_60 Brier      (P(60+) - 1[actual>=60])^2   -> clean-sheet eligibility, the
                  quantity the flat average smears worst
  p_play Brier    (P(plays) - 1[actual>0])^2

Reported over two pools: ALL player-gameweeks with enough history, and the
SQUAD-RELEVANT pool (prior mean minutes >= 45) - genuine options, where a manager
actually feels a rotation call. The half-life that minimises squad-pool minutes
MAE is carried to Gate B; if recent-form does NOT beat crude here, the idea is
dead cheaply.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.minutes import (
    MINUTES_FOR_LONG_APPEARANCE,
    crude_minutes,
    recent_form_minutes,
)
from research.data.fpl_archive import ALL_SEASONS, load_gameweeks
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

FIRST_SCORED_GAMEWEEK = 6          # match the FPL backtest: need some history first
MIN_PRIOR_MATCHES = 5
SQUAD_POOL_MEAN_MINUTES = 45.0     # a genuine option, not a fringe 0-scorer
HALF_LIVES = [1.0, 2.0, 3.0, 4.0, 6.0, 10.0]


def _observations(season):
    """One record per (player, gameweek>=6) with the crude and recent predictions
    and the actual minutes. Minutes history is within-season and strictly prior."""
    frame = load_gameweeks(season)
    # Per player, minutes in chronological gameweek order (0s included: those are
    # the rotation/injury signal). Aggregate double-gameweeks to the GW's total.
    by_player_gw = (frame.groupby(["player_id", "gameweek"])["minutes"].sum()
                    .reset_index().sort_values(["player_id", "gameweek"]))

    records = []
    for player_id, rows in by_player_gw.groupby("player_id"):
        seq, gws = rows["minutes"].tolist(), rows["gameweek"].tolist()
        for i in range(len(seq)):
            if gws[i] < FIRST_SCORED_GAMEWEEK or i < MIN_PRIOR_MATCHES:
                continue
            prior = seq[:i]
            actual = seq[i]
            crude = crude_minutes(sum(prior), len(prior))
            rec = {hl: recent_form_minutes(prior, half_life_matches=hl) for hl in HALF_LIVES}
            records.append({
                "season": season, "player_id": player_id, "gameweek": gws[i],
                "actual": actual, "prior_mean": sum(prior) / len(prior),
                "crude": crude, "recent": rec,
            })
    return records


def _errors(records, model_key, half_life=None):
    """Per-observation minutes error, p_60 error, p_play error for one model."""
    mae, b60, bplay = [], [], []
    for r in records:
        m = r["crude"] if model_key == "crude" else r["recent"][half_life]
        actual = r["actual"]
        mae.append(abs(m["expected_minutes"] - actual))
        b60.append((m["p_60"] - (1.0 if actual >= MINUTES_FOR_LONG_APPEARANCE else 0.0)) ** 2)
        bplay.append((m["p_play"] - (1.0 if actual > 0 else 0.0)) ** 2)
    return np.array(mae), np.array(b60), np.array(bplay)


def _summary(records, label):
    crude_mae, crude_b60, crude_bplay = _errors(records, "crude")
    rows = [{
        "model": "crude (shipped)", "half_life": None,
        "minutes_mae": float(crude_mae.mean()),
        "p60_brier": float(crude_b60.mean()),
        "pplay_brier": float(crude_bplay.mean()),
        "mae_gain_vs_crude": 0.0, "mae_t_p": float("nan"), "mae_wilcoxon_p": float("nan"),
    }]
    for hl in HALF_LIVES:
        mae, b60, bplay = _errors(records, "recent", hl)
        diff = crude_mae - mae      # >0 means recent has lower error
        t_p = float(stats.ttest_rel(mae, crude_mae)[1]) if len(diff) > 2 else float("nan")
        try:
            w_p = float(stats.wilcoxon(mae, crude_mae)[1])
        except ValueError:
            w_p = float("nan")
        rows.append({
            "model": "recent", "half_life": hl,
            "minutes_mae": float(mae.mean()),
            "p60_brier": float(b60.mean()),
            "pplay_brier": float(bplay.mean()),
            "mae_gain_vs_crude": float(diff.mean()),
            "mae_t_p": t_p, "mae_wilcoxon_p": w_p,
        })
    return pd.DataFrame(rows)


def build_report(all_summary, squad_summary, n_all, n_squad, best_hl, run_id):
    def fmt(df):
        lines = ["| model | half-life | minutes MAE | p60 Brier | pplay Brier | "
                 "MAE gain vs crude | t p | Wilcoxon p |",
                 "|---|---|---|---|---|---|---|---|"]
        for _, r in df.iterrows():
            hl = "-" if r["half_life"] is None else ("%.0f" % r["half_life"])
            lines.append("| {0} | {1} | {2:.3f} | {3:.4f} | {4:.4f} | {5:+.3f} | {6} | {7} |".format(
                r["model"], hl, r["minutes_mae"], r["p60_brier"], r["pplay_brier"],
                r["mae_gain_vs_crude"],
                "-" if pd.isna(r["mae_t_p"]) else "%.4f" % r["mae_t_p"],
                "-" if pd.isna(r["mae_wilcoxon_p"]) else "%.4f" % r["mae_wilcoxon_p"]))
        return "\n".join(lines)

    return "\n".join([
        "# Gate A: minutes-prediction accuracy - " + run_id,
        "",
        "Walk-forward WITHIN each of 8 seasons; at gameweek k a model sees only "
        "gameweeks 1..k-1 of that player's minutes. Lower is better. Recent-form "
        "beating crude here is NECESSARY (a better minutes signal exists) but not "
        "sufficient (Gate B decides whether it grows the FPL edge).",
        "",
        "## Squad-relevant pool (prior mean minutes >= {0:.0f}; {1:,} player-gameweeks)".format(
            SQUAD_POOL_MEAN_MINUTES, n_squad),
        "",
        "This is the pool that matters - genuine options a manager weighs, where "
        "rotation calls are made.",
        "",
        fmt(squad_summary),
        "",
        "## All player-gameweeks (>= {0} prior matches; {1:,} obs)".format(MIN_PRIOR_MATCHES, n_all),
        "",
        fmt(all_summary),
        "",
        "## Verdict",
        "",
        _verdict(squad_summary, best_hl),
    ])


def _verdict(squad_summary, best_hl):
    crude_row = squad_summary[squad_summary["model"] == "crude (shipped)"].iloc[0]
    best = squad_summary[squad_summary["half_life"] == best_hl].iloc[0]
    both = best["mae_t_p"] < 0.05 and best["mae_wilcoxon_p"] < 0.05
    better = best["minutes_mae"] < crude_row["minutes_mae"]
    if better and both:
        return (
            "**Recent-form (half-life {0:.0f}) predicts minutes better** on the "
            "squad pool: MAE {1:.3f} vs crude {2:.3f} ({3:+.3f}, significant on both "
            "tests), and p_60 Brier {4:.4f} vs {5:.4f}. Carry half-life {0:.0f} to "
            "Gate B - the FPL-edge test that actually decides.".format(
                best_hl, best["minutes_mae"], crude_row["minutes_mae"],
                best["mae_gain_vs_crude"], best["p60_brier"], crude_row["p60_brier"]))
    if better:
        return (
            "Recent-form (half-life {0:.0f}) has lower minutes MAE ({1:.3f} vs "
            "{2:.3f}) but not significantly on both tests. Weak signal; carry it "
            "to Gate B but expect little.".format(
                best_hl, best["minutes_mae"], crude_row["minutes_mae"]))
    return (
        "**Recent-form does NOT beat the flat average at predicting minutes.** The "
        "shipped crude model is already as good on this pool - the minutes idea is "
        "a null here, cheaply. No need to run Gate B.")


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_minutes_accuracy.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    records = []
    for season in ALL_SEASONS:
        r = _observations(season)
        logger.info("%s: %d player-gameweek observations", season, len(r))
        records.extend(r)

    squad = [r for r in records if r["prior_mean"] >= SQUAD_POOL_MEAN_MINUTES]
    all_summary = _summary(records, "all")
    squad_summary = _summary(squad, "squad")

    # Pick the half-life by squad-pool minutes MAE (the manager-relevant metric).
    recent_only = squad_summary[squad_summary["model"] == "recent"]
    best_hl = float(recent_only.loc[recent_only["minutes_mae"].idxmin(), "half_life"])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(all_summary, squad_summary, len(records), len(squad), best_hl, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("minutes_accuracy_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    print(report)
    print("\nBest half-life (squad-pool MAE): %.0f" % best_hl)
    return best_hl


if __name__ == "__main__":
    main()
