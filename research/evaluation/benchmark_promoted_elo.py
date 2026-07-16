"""Phase 6e Gate A: does a promoted-team Elo prior predict promoted fixtures better?

Promoted teams concede ~+24% / score ~-29% vs the league, but the shipped model
cold-starts every unseen team at the league-average 1500, and with K=20 Elo takes
~15 games to correct - so their fixtures are mis-ranked for much of the first
season. `EloModel(promoted_penalty=p)` starts a team that was not in the previous
season p Elo points lower.

This is the cheap screen (Elo only - it carries ~72% of the ensemble and drives
the grid's clean-sheet region via rescaling). Season-level walk-forward: train on
all prior seasons, predict the eval season with frozen ratings, and score 1X2
prediction on (a) ALL fixtures - a guard that the prior does not hurt overall -
and (b) PROMOTED-team fixtures, where the prior acts. Pick the penalty on
promoted-fixture log loss; the FPL edge is Gate B.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from research.data.fpl_archive import ALL_SEASONS
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.metrics import log_loss, ranked_probability_score
from research.experiments.elo import EloModel

logger = logging.getLogger(__name__)

PENALTIES = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
EPS = 1e-12
_ORDER = {"H": 0, "D": 1, "A": 2}


def _per_obs_log_loss(probs, outcomes):
    idx = np.array([_ORDER[o] for o in outcomes])
    picked = np.clip(probs[np.arange(len(idx)), idx], EPS, 1.0)
    return -np.log(picked)


def run():
    matches = load_understat_matches("EPL")
    seasons = sorted(matches["season"].unique())
    eval_seasons = [s for s in ALL_SEASONS if s in seasons and seasons.index(s) >= 1]

    # promoted teams per eval season: in season S, not in S-1
    promoted = {}
    for s in eval_seasons:
        prev = seasons[seasons.index(s) - 1]
        teams_s = set(matches[matches["season"] == s]["home_team"]) | set(matches[matches["season"] == s]["away_team"])
        teams_prev = set(matches[matches["season"] == prev]["home_team"]) | set(matches[matches["season"] == prev]["away_team"])
        promoted[s] = teams_s - teams_prev

    # collect predictions once per penalty
    cells = []
    per_obs = {}  # penalty -> (all_ll, promoted_mask)
    for penalty in PENALTIES:
        probs_all, outcomes_all, is_promoted = [], [], []
        for s in eval_seasons:
            train = matches[matches["season"] < s]
            test = matches[matches["season"] == s]
            model = EloModel(promoted_penalty=penalty).fit(train)
            probs_all.append(model.predict_proba(test[["home_team", "away_team"]]))
            outcomes_all.extend(test["result"].tolist())
            for h, a in zip(test["home_team"], test["away_team"]):
                is_promoted.append(h in promoted[s] or a in promoted[s])
        probs_all = np.vstack(probs_all)
        outcomes_all = np.array(outcomes_all)
        is_promoted = np.array(is_promoted)

        ll = _per_obs_log_loss(probs_all, outcomes_all)
        prom = is_promoted
        cells.append({
            "penalty": penalty,
            "n_promoted_fixtures": int(prom.sum()),
            "all_log_loss": float(ll.mean()),
            "all_rps": ranked_probability_score(probs_all, outcomes_all),
            "promoted_log_loss": float(ll[prom].mean()),
            "promoted_rps": ranked_probability_score(probs_all[prom], outcomes_all[prom]),
        })
        per_obs[penalty] = (ll, prom)
        logger.info("penalty=%-5.0f promoted_ll=%.4f all_ll=%.4f",
                    penalty, cells[-1]["promoted_log_loss"], cells[-1]["all_log_loss"])

    # significance of each penalty vs 0 on promoted fixtures (paired per fixture)
    base_ll, base_prom = per_obs[0.0]
    for cell in cells:
        ll, prom = per_obs[cell["penalty"]]
        b = base_ll[base_prom]
        c = ll[prom]
        cell["promoted_ll_gain_vs_0"] = float(b.mean() - c.mean())
        if cell["penalty"] == 0.0 or len(c) < 3:
            cell["t_p"], cell["w_p"] = float("nan"), float("nan")
        else:
            cell["t_p"] = float(stats.ttest_rel(c, b)[1])
            try:
                cell["w_p"] = float(stats.wilcoxon(c, b)[1])
            except ValueError:
                cell["w_p"] = float("nan")
    return cells, eval_seasons


def build_report(cells, eval_seasons, run_id):
    default = next(c for c in cells if c["penalty"] == 0.0)
    best = min((c for c in cells if c["penalty"] > 0), key=lambda c: c["promoted_log_loss"])
    both = best["t_p"] < 0.05 and best["w_p"] < 0.05
    better = best["promoted_log_loss"] < default["promoted_log_loss"]
    lines = [
        "# Phase 6e Gate A: promoted-team Elo prior - " + run_id,
        "",
        "Season-level walk-forward on EPL Understat, eval {0}-{1} ({2} promoted-team "
        "fixtures). Elo only. Lower is better; `promoted_log_loss` is the headline "
        "(the fixtures the prior acts on), `all_log_loss` guards against hurting the "
        "rest.".format(eval_seasons[0], eval_seasons[-1], default["n_promoted_fixtures"]),
        "",
        "| promoted_penalty | promoted_log_loss | promoted_rps | all_log_loss | "
        "promoted gain vs 0 | t p | Wilcoxon p |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        tag = " (default)" if c["penalty"] == 0.0 else ""
        lines.append("| {0:.0f}{1} | {2:.4f} | {3:.4f} | {4:.4f} | {5:+.4f} | {6} | {7} |".format(
            c["penalty"], tag, c["promoted_log_loss"], c["promoted_rps"], c["all_log_loss"],
            c["promoted_ll_gain_vs_0"],
            "-" if np.isnan(c["t_p"]) else "%.4f" % c["t_p"],
            "-" if np.isnan(c["w_p"]) else "%.4f" % c["w_p"]))
    lines += [
        "",
        "## Verdict",
        "",
        ("**The prior predicts promoted fixtures better** at penalty={0:.0f}: "
         "promoted log loss {1:.4f} vs default {2:.4f} ({3:+.4f}), {4}. Carry it to "
         "Gate B (the FPL edge).".format(
             best["penalty"], best["promoted_log_loss"], default["promoted_log_loss"],
             best["promoted_ll_gain_vs_0"],
             "significant on both tests" if both else "NOT significant on both - suggestive only")
         if better else
         "**The prior does not improve promoted-fixture prediction** - the cold-start "
         "cost is already small at season level or the penalty is mis-sized. Null."),
    ]
    return "\n".join(lines), best["penalty"]


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_promoted_elo.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    cells, eval_seasons = run()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report, best_penalty = build_report(cells, eval_seasons, run_id)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / ("promoted_elo_report_" + run_id + ".md")).write_text(report, encoding="utf-8")
    print(report)
    print("\nBest penalty (promoted-fixture log loss): %.0f" % best_penalty)
    return best_penalty


if __name__ == "__main__":
    main()
