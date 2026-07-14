"""Phase 6a: does tuning the training WINDOW and time-decay grow the edge?

A deep-research pass (2026-07) found that, for match prediction, the choice of
goal-model family (Poisson vs Dixon-Coles vs bivariate) is out-of-sample noise,
but tuning the training-data window (~4 seasons) plus Dixon-Coles time-decay
(xi ~= 0.001) moved RPS ~10x more than the entire family spread. That is the
one team-model lever the research rated cheap-and-worthwhile.

Our prior tuning pass (tuning_report_20260710T051002Z) already tested xi on
{0.0, 0.002, 0.0065, 0.012} and found a NULL on 1X2 log loss - but that test
never varied the WINDOW (the whole harness is expanding-window), never tested
xi ~= 0.001, and measured 1X2 log loss, where Elo carries ~72% of the ensemble
weight and ignores xi entirely. So the prior null does NOT settle this.

The channel that actually feeds our FPL edge is the scoreline GRID - clean-sheet
probability and the goals-conceded distribution - and that grid comes ONLY from
the two goal models (Elo has no grid). This screen therefore isolates exactly
that channel, on the two goal models, where window and xi genuinely act.

============================ PRE-REGISTRATION ============================
Written before the run. The literature-default cell is inside the grid, so the
experiment can (and is expected to be able to) conclude "the default is already
near-optimal" - a null is a first-class outcome here.

Levers:
    window (max_train_seasons): {expanding, 2, 3, 4, 5}
    xi (per-day decay, DC only): {0.0, 0.0005, 0.001, 0.002, 0.0065}
    Default cell   = (expanding, xi=0.0065).   Research cell = (4 seasons, 0.001).
    PoissonXG has no time-decay, so only its window varies.

Screening endpoint (THIS script, cheap - goal models only, no Elo/ensemble):
    Walk-forward on EPL Understat, evaluated on the seasons where even a 5-season
    window has full history, so every cell is scored on the IDENTICAL matches.
    Per (window, xi) cell, per model:
      - WDL:         log_loss, RPS                    (the research metric)
      - CLEAN SHEET: log_loss + Brier of P(clean sheet) read off the score grid,
                     over both sides of every match   (the FPL-relevant channel).
                     Home CS <=> away scored 0 <=> sum(grid[:, 0]).
    SCREEN PASS = a cell beats its model's default cell on clean-sheet log loss
    AND does not worsen WDL log loss, on the same matches. The best passing cell's
    edge over the default is significance-tested (paired t + Wilcoxon) on the
    per-observation clean-sheet loss.

Primary endpoint (Stage 2, a SEPARATE script, run only for screen-passing cells):
    the FPL GBP100m-squad + captain gain over player_ppg - identical to Phase 5e's
    pre-specified primary. A new (window, xi) REPLACES the default ONLY if it beats
    the default configuration on that gain, significant on BOTH paired t and
    Wilcoxon (p<0.05) after Holm correction across the carried cells, and
    non-negative in per-season replication. Otherwise: record the null, keep the
    literature default.

Honesty guard (this project's measurement-artifact history is littered with
metrics that only ever flattered us): the default cell is scored identically to
every other, multiplicity is Holm-corrected, and the null ships. A lever that
only ever helps is treated as a bug in the metric until proven otherwise.
========================================================================
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.metrics import log_loss, ranked_probability_score
from research.experiments.dixon_coles_xg import DixonColesXGModel
from research.experiments.poisson_xg import PoissonXGModel

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUE = "EPL"

# The grid. `None` window = expanding (all prior seasons) = today's behaviour.
WINDOWS = [None, 2, 3, 4, 5]
XIS = [0.0, 0.0005, 0.001, 0.002, 0.0065]
XI_DEFAULT = 0.0065          # the literature / current default
MAX_WINDOW = 5               # longest finite window -> min history needed to compare fairly

EPS = 1e-12


def _windowed_train(matches: pd.DataFrame, season: str, max_train_seasons):
    """Matches strictly before `season`, optionally truncated to the most recent
    `max_train_seasons` seasons (None = keep all = expanding window)."""
    prior = matches[matches["season"] < season]
    if max_train_seasons is None:
        return prior
    keep = sorted(prior["season"].unique())[-max_train_seasons:]
    return prior[prior["season"].isin(keep)]


def _clean_sheet_probs(model, home_team: str, away_team: str):
    """(P(home keeps clean sheet), P(away keeps clean sheet)) from the score grid.

    Home keeps a clean sheet iff the away team scores 0 -> the away=0 column,
    grid[:, 0]. Grid is renormalised for the tail truncation first, exactly as
    predict_proba does, so the two probabilities are on the same footing."""
    grid = np.asarray(model.score_grid(home_team, away_team), dtype=float)
    grid = grid / grid.sum()
    return float(grid[:, 0].sum()), float(grid[0, :].sum())


def run_cell(matches, build, max_train_seasons, eval_seasons):
    """Walk-forward one (model, window) configuration over `eval_seasons`.

    Returns per-observation arrays so any cell can be paired against the default
    on the identical matches:
      wdl_probs   (n_matches, 3)   H/D/A probabilities
      wdl_outcome (n_matches,)     'H'/'D'/'A'
      cs_prob     (2*n_matches,)   P(clean sheet), home side then away side
      cs_outcome  (2*n_matches,)   1.0 if that side kept a clean sheet else 0.0
    """
    wdl_probs, wdl_outcome = [], []
    cs_prob, cs_outcome = [], []

    for season in eval_seasons:
        train = _windowed_train(matches, season, max_train_seasons)
        test = matches[matches["season"] == season]
        model = build().fit(train)

        wdl_probs.append(model.predict_proba(test[["home_team", "away_team"]]))
        wdl_outcome.extend(test["result"].tolist())

        for row in test.itertuples():
            p_home_cs, p_away_cs = _clean_sheet_probs(model, row.home_team, row.away_team)
            cs_prob.extend([p_home_cs, p_away_cs])
            cs_outcome.extend([float(row.away_goals == 0), float(row.home_goals == 0)])

    return {
        "wdl_probs": np.vstack(wdl_probs),
        "wdl_outcome": np.array(wdl_outcome),
        "cs_prob": np.array(cs_prob),
        "cs_outcome": np.array(cs_outcome),
    }


def _binary_log_loss_per_obs(prob, outcome):
    """Per-observation binary log loss for the clean-sheet head."""
    prob = np.clip(prob, EPS, 1.0 - EPS)
    return -(outcome * np.log(prob) + (1.0 - outcome) * np.log(1.0 - prob))


def _wdl_log_loss_per_obs(probs, outcome):
    order = {"H": 0, "D": 1, "A": 2}
    idx = np.array([order[o] for o in outcome])
    picked = np.clip(probs[np.arange(len(idx)), idx], EPS, 1.0)
    return -np.log(picked)


def _cell_metrics(arrays):
    cs_ll = _binary_log_loss_per_obs(arrays["cs_prob"], arrays["cs_outcome"])
    return {
        "wdl_log_loss": log_loss(arrays["wdl_probs"], arrays["wdl_outcome"]),
        "wdl_rps": ranked_probability_score(arrays["wdl_probs"], arrays["wdl_outcome"]),
        "cs_log_loss": float(cs_ll.mean()),
        "cs_brier": float(np.mean((arrays["cs_prob"] - arrays["cs_outcome"]) ** 2)),
    }


def _paired_p(challenger_loss, default_loss):
    """Paired significance of a per-observation loss reduction (lower is better).
    Returns (mean_reduction, t_p, wilcoxon_p). Positive reduction = challenger
    better."""
    diff = default_loss - challenger_loss    # >0 means challenger has lower loss
    mean = float(diff.mean())
    t_p = float(stats.ttest_rel(challenger_loss, default_loss)[1]) if len(diff) > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(challenger_loss, default_loss)[1])
    except ValueError:
        w_p = float("nan")   # all-zero differences
    return mean, t_p, w_p


def evaluate_model(name, build, has_xi, matches, eval_seasons):
    """Run every (window, xi) cell for one model; return a list of cell dicts and
    the default cell's per-obs losses (for pairing)."""
    cells = []
    default_arrays = None
    xis = XIS if has_xi else [None]

    for window in WINDOWS:
        for xi in xis:
            def make():
                return build(xi=xi) if has_xi else build()
            arrays = run_cell(matches, make, window, eval_seasons)
            metrics = _cell_metrics(arrays)
            is_default = (window is None) and ((xi == XI_DEFAULT) or not has_xi)
            if is_default:
                default_arrays = arrays
            cells.append({
                "model": name,
                "window": "expanding" if window is None else window,
                "xi": xi if has_xi else "n/a",
                "is_default": is_default,
                "_arrays": arrays,
                **metrics,
            })
            logger.info("%s window=%s xi=%s  cs_ll=%.4f wdl_ll=%.4f",
                        name, window, xi, metrics["cs_log_loss"], metrics["wdl_log_loss"])

    # Significance of every cell vs the default, on identical matches.
    default_cs_ll = _binary_log_loss_per_obs(
        default_arrays["cs_prob"], default_arrays["cs_outcome"])
    default_wdl_ll = _wdl_log_loss_per_obs(
        default_arrays["wdl_probs"], default_arrays["wdl_outcome"])
    default_metrics = _cell_metrics(default_arrays)

    for cell in cells:
        cs_ll = _binary_log_loss_per_obs(
            cell["_arrays"]["cs_prob"], cell["_arrays"]["cs_outcome"])
        wdl_ll = _wdl_log_loss_per_obs(
            cell["_arrays"]["wdl_probs"], cell["_arrays"]["wdl_outcome"])
        cs_gain, cs_t, cs_w = _paired_p(cs_ll, default_cs_ll)
        cell["cs_ll_gain_vs_default"] = cs_gain
        cell["cs_t_p"] = cs_t
        cell["cs_wilcoxon_p"] = cs_w
        # SCREEN PASS: strictly better clean-sheet log loss, and WDL log loss not
        # worsened (within a tiny tolerance) vs the default.
        cell["screen_pass"] = bool(
            cell["cs_log_loss"] < default_metrics["cs_log_loss"]
            and cell["wdl_log_loss"] <= default_metrics["wdl_log_loss"] + 1e-6
        )
        del cell["_arrays"]

    return cells, default_metrics


def build_report(all_cells, default_metrics_by_model, run_config, run_id):
    lines = [
        "# Phase 6a: Training-window x time-decay screen - " + run_id,
        "",
        "League: EPL (Understat). Walk-forward, evaluated on "
        + "{0} to {1} ({2} seasons, {3} matches) so every window setting (up to {4} "
        "seasons) has full history and all cells are scored on the IDENTICAL "
        "matches.".format(
            run_config["eval_seasons"][0], run_config["eval_seasons"][-1],
            len(run_config["eval_seasons"]), run_config["n_matches"], MAX_WINDOW),
        "",
        "**This is the cheap SCREEN (Stage 1), goal models only.** It isolates the "
        "scoreline-grid channel that feeds the FPL edge - clean-sheet probability - "
        "which the prior expanding-window, 1X2-only tuning pass never tested. "
        "Screen-passing cells (if any) go to the Stage 2 FPL pts/GW primary; "
        "nothing here changes the shipped model on its own.",
        "",
        "Lower is better for every column. `cs_log_loss` is the headline (the "
        "FPL channel); `wdl_log_loss`/`wdl_rps` guard against buying clean-sheet "
        "accuracy by wrecking the 1X2 forecast. The **default** cell (expanding "
        "window, xi=0.0065) is marked - it is what ships today.",
        "",
    ]

    for model_name in sorted(set(c["model"] for c in all_cells)):
        model_cells = [c for c in all_cells if c["model"] == model_name]
        dm = default_metrics_by_model[model_name]
        lines += [
            "## " + model_name,
            "",
            "Default cell: cs_log_loss **{0:.4f}**, wdl_log_loss {1:.4f}, "
            "wdl_rps {2:.4f}.".format(dm["cs_log_loss"], dm["wdl_log_loss"], dm["wdl_rps"]),
            "",
            "| window | xi | cs_log_loss | cs_brier | wdl_log_loss | wdl_rps | "
            "cs gain vs default | t p | Wilcoxon p | screen |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for c in sorted(model_cells, key=lambda x: x["cs_log_loss"]):
            tag = " (default)" if c["is_default"] else (" **PASS**" if c["screen_pass"] else "")
            lines.append(
                "| {0}{1} | {2} | {3:.4f} | {4:.4f} | {5:.4f} | {6:.4f} | "
                "{7:+.5f} | {8:.4f} | {9:.4f} | {10} |".format(
                    c["window"], tag, c["xi"], c["cs_log_loss"], c["cs_brier"],
                    c["wdl_log_loss"], c["wdl_rps"], c["cs_ll_gain_vs_default"],
                    c["cs_t_p"], c["cs_wilcoxon_p"],
                    "yes" if c["screen_pass"] else ""))
        lines.append("")

        passes = [c for c in model_cells if c["screen_pass"]]
        if not passes:
            lines += [
                "**No cell beats the default on the clean-sheet channel without "
                "hurting the 1X2 forecast.** For " + model_name + ", the literature "
                "default (expanding window, xi=0.0065) is already at or beyond the "
                "best of the grid - a clean null for this lever.",
                "",
            ]
        else:
            best = min(passes, key=lambda x: x["cs_log_loss"])
            both = best["cs_t_p"] < 0.05 and best["cs_wilcoxon_p"] < 0.05
            lines += [
                "Best screen-passing cell: **window={0}, xi={1}** - cs_log_loss "
                "{2:.4f} vs default {3:.4f} (gain {4:+.5f}/obs, t p={5:.4f}, "
                "Wilcoxon p={6:.4f}). {7}".format(
                    best["window"], best["xi"], best["cs_log_loss"],
                    dm["cs_log_loss"], best["cs_ll_gain_vs_default"],
                    best["cs_t_p"], best["cs_wilcoxon_p"],
                    "Significant on BOTH tests -> carry to Stage 2 FPL primary."
                    if both else
                    "NOT significant on both tests -> suggestive only; carry the "
                    "top 1-2 cells to Stage 2 but expect a likely null."),
                "",
            ]

    lines += [
        "## What happens next",
        "",
        "Per the pre-registration, screen-passing cells are carried to the Stage 2 "
        "FPL primary endpoint (GBP100m squad + captain gain over player_ppg, Holm-"
        "corrected). The shipped model changes only if a cell wins THERE. If no "
        "cell passes this screen, the expanding-window / xi=0.0065 default stands "
        "and the window/decay lever is recorded as a null - the honest, expected "
        "outcome given the prior 1X2 tuning null and the low FPL predictability "
        "ceiling.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_window_decay.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    matches = load_understat_matches(UNDERSTAT_LEAGUE)
    all_seasons = sorted(matches["season"].unique())
    # Fair-comparison window: every cell needs MAX_WINDOW seasons of history
    # before the first evaluated season, so all cells score the same matches.
    eval_seasons = all_seasons[MAX_WINDOW:]
    eval_matches = matches[matches["season"].isin(eval_seasons)]
    logger.info("Loaded %d matches; evaluating on %s (%d matches)",
                len(matches), eval_seasons, len(eval_matches))

    model_specs = [
        ("dixon_coles_xg", DixonColesXGModel, True),
        ("poisson_xg", PoissonXGModel, False),
    ]

    all_cells = []
    default_metrics_by_model = {}
    for name, build, has_xi in model_specs:
        cells, default_metrics = evaluate_model(name, build, has_xi, matches, eval_seasons)
        all_cells.extend(cells)
        default_metrics_by_model[name] = default_metrics

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_config = {
        "run_id": run_id,
        "league": UNDERSTAT_LEAGUE,
        "eval_seasons": eval_seasons,
        "n_matches": int(len(eval_matches)),
        "windows": ["expanding" if w is None else w for w in WINDOWS],
        "xis": XIS,
        "default_cell": {"window": "expanding", "xi": XI_DEFAULT},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(all_cells, default_metrics_by_model, run_config, run_id)
    (RESULTS_DIR / ("window_decay_report_" + run_id + ".md")).write_text(
        report, encoding="utf-8")
    (RESULTS_DIR / ("window_decay_cells_" + run_id + ".json")).write_text(
        json.dumps({"config": run_config, "cells": all_cells}, indent=2), encoding="utf-8")

    print(report)
    print("\nArtifacts written to " + str(RESULTS_DIR))
    return all_cells


if __name__ == "__main__":
    main()
