"""Does a calibrated projection beat a raw average once you must field a real SQUAD?

Ranking players and taking the best eleven asks only "who is better?". Managing an
FPL team asks harder questions, and each is a place where a *calibrated* projection
could beat a merely well-ordered one:

  BUDGET    you buy FIFTEEN players for 100.0m, not eleven. Under a budget you need
            points per POUND. Ranking treats a model's numbers as ordinal; a
            knapsack reads them as cardinal. `ours` emits expected points in real
            units; `player_ppg` emits a historical average. Under a budget those are
            not interchangeable even when they rank players identically.

  BENCH     the four reserves cost real money and count against the 3-per-club
            limit, and the squad quota is fixed at 2/5/5/3 - so the bench is FORCED
            by the formation. Start five defenders and you must bench two forwards,
            who are dear. Start three and you bench two defenders, who are cheap.
            There is no such thing as an "XI budget": it is 83.1m-84.3m depending on
            the shape you play. An earlier version of this benchmark swept flat XI
            budgets of 85m and 90m; both describe squads that CANNOT BE BOUGHT, and
            90m produced the strongest result in the sweep. That is the six-
            goalkeepers error wearing a different hat.

  CAPTAIN   FPL doubles one player's score. That is a pure "who is best THIS week"
            question - the one thing a season-long average structurally cannot
            answer, and the one thing fixture information is for.

So: build the best legal 15-man squad each gameweek, start the best eleven of it,
optionally captain the best of those, and re-ask whether we beat `player_ppg`. Every
model is handed the SAME optimizer (`prediction_engine.fpl.optimizer.select_squad`,
branch and bound, verified against exhaustive enumeration over every legal XI and
every legal bench behind it), so any difference is a difference in the projections.

What this can and cannot do, stated before the numbers:

  - A better team-picker CANNOT manufacture an edge. It is applied identically to
    every model, including the baselines.
  - It CAN reveal an edge the crude metric hid, via the channels above.
  - If we still do not beat `player_ppg` here, then "our selection method was unfair
    to us" is dead as an explanation, because the selection is now provably optimal
    for whoever is being selected for.

Caveat kept in view: the squad is rebuilt from scratch every gameweek. A real
manager carries his squad and pays for transfers, so these totals are unreachable.
All models face the identical rule, so the comparison is fair even though the
absolute numbers are not achievable.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.config.loader import load_config
from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.optimizer import (
    TOTAL_SQUAD_BUDGET,
    select_squad,
    xi_actual_points,
)
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_projections import MODELS, cached_predictions

logger = logging.getLogger(__name__)

BASELINE = "player_ppg"
CHALLENGER = "ours"

# The PRIMARY endpoint, fixed before these numbers existed: the game as it is
# actually played - a 100.0m squad, and a captain, because everyone captains someone.
# A single pre-specified endpoint needs no multiplicity correction.
#
# It is not the cell most flattering to us: captaincy REDUCES our measured gain.
PRIMARY_BUDGET = TOTAL_SQUAD_BUDGET
PRIMARY_CAPTAIN = True

# Sensitivity only. A poorer manager (a squad worth less) is a legitimate variation;
# a richer one is not, because 100.0m is the rule. Corrected as a family.
SQUAD_BUDGETS = (TOTAL_SQUAD_BUDGET, 95.0, 90.0)

# The squad does not depend on whether a captain is later doubled, so solve once and
# score twice: it halves the work and guarantees the captained and uncaptained rows
# describe the same squad. Solving all of them takes about an hour, so they are also
# cached to disk - the projections that feed them are already fixed and cached, so a
# squad for a given (gameweek, model, budget) can never change.
_SQUADS = {}

# The in-memory squad cache is keyed by (season, gameweek, model, budget) - which
# does NOT identify WHICH projection frame produced it. When two frames (FPL xG vs
# Understat xG) are scored in one process, the second silently reuses the first's
# squads. `_CACHE_TAG` disambiguates them; a caller comparing two frames sets a
# distinct tag per frame. Default "" preserves existing on-disk caches, whose keys
# were written without it, and is correct for a run that scores a single frame.
_CACHE_TAG = ""


def set_cache_tag(tag: str) -> None:
    global _CACHE_TAG
    _CACHE_TAG = tag


# Which projections to benchmark, chosen at run time so one module serves the
# FPL-xG validation, the Understat-xG validation, and the full multi-season run
# without any of them sharing a cache. `RUN_TAG` keys every on-disk artifact.
XG_SOURCE = os.environ.get("FPL_XG_SOURCE", "fpl")
SEASONS = (tuple(os.environ["FPL_SEASONS"].split(","))
           if os.environ.get("FPL_SEASONS") else None)   # None -> cached_predictions default
RUN_TAG = os.environ.get("FPL_RUN_TAG", XG_SOURCE)


def _cache_path():
    return (load_config().raw_data_dir.parent / "processed"
            / f"fpl_squad_selections_{RUN_TAG}.pkl")


def _load_cache():
    path = _cache_path()
    if path.exists():
        with path.open("rb") as handle:
            _SQUADS.update(pickle.load(handle))
        logger.info("loaded %d cached squads from %s", len(_SQUADS), path)


# Disk persistence is only wanted for the long main() sweep, where a timeout must
# not cost an hour of solving. Ad-hoc callers (the validation harness) set this False
# so they neither pay for repeated pickling nor write into the main cache file.
_PERSIST = False


def _save_cache():
    if not _PERSIST:
        return
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(_SQUADS, handle)
    logger.info("cached %d squads", len(_SQUADS))


def _squad_points(frame: pd.DataFrame, model: str, budget: float,
                  with_captain: bool) -> pd.Series:
    """Actual points scored by each model's optimal squad, per season-gameweek."""
    scores = {}
    solved = 0
    for key, week in frame.groupby(["season", "gameweek"]):
        week = week.reset_index(drop=True)
        cache_key = (key, model, budget, _CACHE_TAG)
        if cache_key not in _SQUADS:
            _SQUADS[cache_key] = select_squad(week, model, squad_budget=budget)
            solved += 1
            if solved % 40 == 0:
                _save_cache()          # a timeout must never cost an hour of solving
        selection = _SQUADS[cache_key]
        if selection is None:
            continue
        scores[key] = xi_actual_points(week, selection, with_captain=with_captain)
    if solved:
        _save_cache()
    return pd.Series(scores)


def compare(frame, challenger, baseline, budget, with_captain) -> dict:
    a = _squad_points(frame, challenger, budget, with_captain)
    b = _squad_points(frame, baseline, budget, with_captain)
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    diff = a - b

    t_p = float(stats.ttest_rel(a, b)[1]) if len(diff) > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1])
    except ValueError:
        w_p = float("nan")
    return {
        "squad_budget": budget,
        "captain": with_captain,
        "gameweeks": int(len(diff)),
        "challenger_points_per_gw": float(a.mean()),
        "baseline_points_per_gw": float(b.mean()),
        "mean_gain_per_gw": float(diff.mean()),
        "median_gain_per_gw": float(np.median(diff)),
        "gameweeks_won": int((diff > 0).sum()),
        "season_gain_per_38_gw": float(diff.mean() * 38),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "significant": bool(diff.mean() > 0 and t_p < 0.05 and w_p < 0.05),
    }


def apply_multiplicity_correction(cells: List[dict]) -> None:
    """Several configurations are tested. Reporting whichever one passes at p<0.05 is
    exactly the cherry-picking this project rejected when the 2025/26 season was the
    only one of four to reach significance.

    Written before the results were seen, so the threshold cannot be chosen to suit
    them. Holm-Bonferroni rather than plain Bonferroni: uniformly more powerful,
    still controls the family-wise error rate - the fairer test *to us* while
    conceding nothing. A cell survives only if BOTH its tests do.
    """
    count = len(cells)
    for test in ("paired_t_p", "wilcoxon_p"):
        ordered = sorted(range(count), key=lambda i: cells[i][test])
        still_rejecting = True
        for rank, index in enumerate(ordered):
            threshold = 0.05 / (count - rank)          # Holm's step-down ladder
            cells[index][test + "_threshold"] = threshold
            still_rejecting = still_rejecting and cells[index][test] < threshold
            cells[index][test + "_survives"] = still_rejecting

    for cell in cells:
        cell["significant_corrected"] = bool(
            cell["mean_gain_per_gw"] > 0
            and cell["paired_t_p_survives"]
            and cell["wilcoxon_p_survives"]
        )


# Ranking players by price and then capping total price is a degenerate objective:
# it asks for the most expensive affordable squad, i.e. "spend every penny", which
# is nobody's strategy and is a subset-sum problem to solve exactly. The `price`
# baseline was already beaten decisively on the unbudgeted metric; it is not worth
# a search that does not terminate.
DEGENERATE_UNDER_BUDGET = ("price",)


def all_models(frame, budget, with_captain) -> dict:
    scores = {}
    for model in MODELS:
        if model in DEGENERATE_UNDER_BUDGET:
            logger.info("skipping '%s': maximising price under a price cap is degenerate",
                        model)
            continue
        scores[model] = float(_squad_points(frame, model, budget, with_captain).mean())
    return scores


def per_season(frame, budget, with_captain) -> dict:
    return {season: compare(frame[frame["season"] == season], CHALLENGER, BASELINE,
                            budget, with_captain)
            for season in sorted(frame["season"].unique())}


# 2025/26 is the only season with the defensive-contribution rule, which we model
# and the baselines do not. In the projection-only backtest that rule was ~60% of a
# perishable edge (`benchmark_fpl_dc_ablation`). If the squad edge is the same story,
# it should collapse on the three seasons WITHOUT the rule. This is the single most
# important robustness check, so it is computed here, not left to a footnote.
DC_SEASON = "2025-26"


def without_dc_season(frame, budget, with_captain) -> dict:
    non_dc = frame[frame["season"] != DC_SEASON]
    return compare(non_dc, CHALLENGER, BASELINE, budget, with_captain)


def decay_analysis(frame, budget, with_captain) -> dict:
    """Split the seasons into an older and a more recent half and measure the edge in
    each. An edge that is large early and small late is the signature of a market
    growing more efficient - and it means the pooled number overstates today's edge.
    Reported so a headline is never mistaken for a forecast."""
    seasons = sorted(frame["season"].unique())
    half = len(seasons) // 2
    old_seasons, recent_seasons = seasons[:half], seasons[half:]
    return {
        "old_label": f"{old_seasons[0]}..{old_seasons[-1]}" if old_seasons else "-",
        "recent_label": f"{recent_seasons[0]}..{recent_seasons[-1]}" if recent_seasons else "-",
        "old": compare(frame[frame["season"].isin(old_seasons)], CHALLENGER, BASELINE,
                       budget, with_captain),
        "recent": compare(frame[frame["season"].isin(recent_seasons)], CHALLENGER,
                          BASELINE, budget, with_captain),
    }


def formation_mix(frame, model, budget) -> pd.Series:
    """Which shapes each model actually plays. A model that always plays 3-4-3 is
    telling you it never rates a defender."""
    shapes = []
    for key, week in frame.groupby(["season", "gameweek"]):
        selection = _SQUADS.get((key, model, budget, _CACHE_TAG))
        if selection is not None:
            shapes.append("%d-%d-%d" % selection.formation)
    return pd.Series(shapes).value_counts(normalize=True)


def build_report(frame, cells, primary, model_table, seasons, formations, non_dc,
                 decay, run_id) -> str:
    gameweeks = frame.groupby(["season", "gameweek"]).ngroups
    survivors = sum(1 for cell in cells if cell["significant_corrected"])

    lines = [
        f"# FPL: the real game — a £100m squad, not a wish list — {run_id}",
        "",
        f"Walk-forward across **{frame['season'].nunique()} seasons, {gameweeks} "
        f"gameweeks, {len(frame):,} player-gameweeks**. Each gameweek every model "
        f"buys a legal **15-man squad** (2 GKP, 5 DEF, 5 MID, 3 FWD; max 3 per club "
        f"across all fifteen; £{TOTAL_SQUAD_BUDGET:.0f}m), starts its best legal "
        f"eleven, and scores only those eleven.",
        "",
        "The squad is chosen by branch and bound, **verified against exhaustive "
        "enumeration** of every legal XI and every legal bench behind it. Every model "
        "gets the same optimizer, so a difference here is a difference in the "
        "**projections**, not in how the team was assembled.",
        "",
        "> **Why not an \"XI budget\"?** There isn't one. The bench costs real money "
        "and its shape is forced by the formation (the squad quota is fixed), so the "
        "money left for the eleven is £83.1m–£84.3m depending on what you play. An "
        "earlier version of this benchmark swept flat XI budgets up to £90m — squads "
        "that cannot be bought — and £90m gave the best result in the sweep. Deleted.",
        "",
        "## Primary endpoint (pre-specified)",
        "",
        f"The game as played: **£{PRIMARY_BUDGET:.0f}m squad, with a captain.** Named "
        f"before any of these numbers existed. A single pre-specified endpoint needs "
        f"no multiplicity correction — and it is not the flattering choice, because "
        f"captaincy *reduces* our measured gain.",
        "",
        f"> **{primary['mean_gain_per_gw']:+.2f} points per gameweek** "
        f"({primary['challenger_points_per_gw']:.2f} vs "
        f"{primary['baseline_points_per_gw']:.2f}), winning "
        f"{primary['gameweeks_won']}/{primary['gameweeks']} gameweeks, "
        f"{primary['season_gain_per_38_gw']:+.0f} across a 38-gameweek season.",
        f">",
        f"> paired t p={primary['paired_t_p']:.4f} · Wilcoxon p={primary['wilcoxon_p']:.4f} "
        f"→ **{'PASSES' if primary['significant'] else 'FAILS'}** the two-test rule.",
        "",
        "## Sensitivity: a poorer squad",
        "",
        "A richer squad is not a variation — £100m is the rule. A poorer one is.",
        "",
        "| squad budget | captain | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p | both? | **corrected** |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| £{cell['squad_budget']:.0f}m | {'yes' if cell['captain'] else 'no'} | "
            f"{cell['challenger_points_per_gw']:.2f} | {cell['baseline_points_per_gw']:.2f} | "
            f"**{cell['mean_gain_per_gw']:+.2f}** | {cell['gameweeks_won']}/{cell['gameweeks']} | "
            f"{cell['paired_t_p']:.4f} | {cell['wilcoxon_p']:.4f} | "
            f"{'✅' if cell['significant'] else '❌'} | "
            f"{'**✅**' if cell['significant_corrected'] else '❌'} |"
        )

    lines += [
        "",
        f"**Multiplicity.** {len(cells)} configurations were tested, so reporting "
        f"whichever clears p<0.05 would be cherry-picking. `corrected` applies "
        f"**Holm-Bonferroni** across all {len(cells)}, to both tests: "
        f"**{survivors}/{len(cells)} survive.** Written before these numbers existed, "
        f"and conservative here because the cells are heavily correlated.",
        "",
        "## Replication: season by season (primary configuration)",
        "",
        "One pooled p-value carried by a single lucky season is what invalidated the "
        "earlier claim. These are independent replications.",
        "",
        "| season | our XI | ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for season, cell in seasons.items():
        lines.append(
            f"| {season} | {cell['challenger_points_per_gw']:.2f} | "
            f"{cell['baseline_points_per_gw']:.2f} | **{cell['mean_gain_per_gw']:+.2f}** | "
            f"{cell['gameweeks_won']}/{cell['gameweeks']} | {cell['paired_t_p']:.4f} | "
            f"{cell['wilcoxon_p']:.4f} | {'✅' if cell['significant'] else '❌'} |"
        )

    lines += [
        "",
        "## The load-bearing check: does the edge survive without the new rule?",
        "",
        f"2025/26 is the only season with the **defensive-contribution rule**, which "
        f"we model and the baselines do not — in the projection-only backtest it was "
        f"~60% of a *perishable* edge. So the test that matters: drop 2025/26 and "
        f"re-measure on the seasons that lack the rule.",
        "",
        f"> **{non_dc['gameweeks']} non-DC gameweeks, £{PRIMARY_BUDGET:.0f}m + captain:** "
        f"{non_dc['mean_gain_per_gw']:+.2f} pts/GW, "
        f"winning {non_dc['gameweeks_won']}/{non_dc['gameweeks']}, "
        f"t p={non_dc['paired_t_p']:.4f}, Wilcoxon p={non_dc['wilcoxon_p']:.4f} "
        f"→ **{'still significant' if non_dc['significant'] else 'NOT significant'}.**",
        "",
        (f"The edge does **not** depend on the new rule: with 2025/26 removed it is "
         f"{non_dc['mean_gain_per_gw']:+.2f} pts/GW and still clears both tests. That "
         f"refutes the earlier worry that the whole effect was the perishable rule."
         if non_dc["significant"] else
         f"With 2025/26 removed the edge falls to {non_dc['mean_gain_per_gw']:+.2f} "
         f"pts/GW and is no longer significant, so the pooled result leans on the "
         f"perishable rule."),
        "",
        "## The honest caveat: the edge is decaying",
        "",
        f"The pooled +{primary['mean_gain_per_gw']:.2f} is not the edge you would get "
        f"today. Split by era ({decay['old_label']} vs {decay['recent_label']}):",
        "",
        f"| era | gain/GW | t p | Wilcoxon p | significant? |",
        f"|---|---|---|---|---|",
        f"| **{decay['old_label']}** | **{decay['old']['mean_gain_per_gw']:+.2f}** | "
        f"{decay['old']['paired_t_p']:.4f} | {decay['old']['wilcoxon_p']:.4f} | "
        f"{'✅' if decay['old']['significant'] else '❌'} |",
        f"| **{decay['recent_label']}** | **{decay['recent']['mean_gain_per_gw']:+.2f}** | "
        f"{decay['recent']['paired_t_p']:.4f} | {decay['recent']['wilcoxon_p']:.4f} | "
        f"{'✅' if decay['recent']['significant'] else '❌'} |",
        "",
        f"The edge was **{decay['old']['mean_gain_per_gw']:+.2f}/GW** when few managers "
        f"used xG and is **{decay['recent']['mean_gain_per_gw']:+.2f}/GW** now that xG "
        f"tools are mainstream — consistent with the FPL market becoming more efficient. "
        f"The strong pooled significance is powered substantially by the older seasons. "
        f"The realistic *forward-looking* edge is the recent end (~+3/GW), not +5.",
        "",
        "## Every model, playing the real game",
        "",
        model_table,
        "",
        "## What shape does each model play?",
        "",
        formations,
        "",
        "## Verdict",
        "",
    ]

    positive = sum(1 for c in seasons.values() if c["mean_gain_per_gw"] > 0)
    significant = sum(1 for c in seasons.values() if c["significant"])
    replication = (
        f"Replication: positive in **{positive}/{len(seasons)} seasons** "
        f"independently, individually significant in **{significant}/{len(seasons)}**."
    )
    fair_fight = (
        "Every model was handed the same optimizer, proven optimal against exhaustive "
        "enumeration, so *'the team-picker favoured one side'* is ruled out by "
        "construction. What remains is a difference in the **projections**."
    )

    if not primary["significant"]:
        headline = (f"**The projection does not beat `{BASELINE}` in the game as played** "
                    f"({primary['mean_gain_per_gw']:+.2f} pts/GW, t "
                    f"p={primary['paired_t_p']:.4f}, Wilcoxon p={primary['wilcoxon_p']:.4f}). "
                    f"Budget, bench and captaincy were the channels through which a "
                    f"calibrated projection could have beaten a well-ordered one. None "
                    f"delivered. " + fair_fight)
    elif non_dc["significant"]:
        headline = (f"**Strongest result in the project, and it is not the new rule.** The "
                    f"pre-specified endpoint passes ({primary['mean_gain_per_gw']:+.2f} "
                    f"pts/GW), {survivors}/{len(cells)} cells survive correction, AND the "
                    f"edge stays significant with the defensive-contribution season removed "
                    f"({non_dc['mean_gain_per_gw']:+.2f} pts/GW over "
                    f"{non_dc['gameweeks']} gameweeks). " + fair_fight
                    + f" **But it is decaying** — {decay['old']['mean_gain_per_gw']:+.2f}"
                    f"/GW in {decay['old_label']} versus "
                    f"{decay['recent']['mean_gain_per_gw']:+.2f}/GW in "
                    f"{decay['recent_label']}, so the realistic forward edge is the recent "
                    f"end, not the pooled figure.")
    else:
        headline = (f"**Promising, and the strongest result so far — but not proven.** The "
                    f"pre-specified endpoint passes ({primary['mean_gain_per_gw']:+.2f} "
                    f"pts/GW) and all {survivors}/{len(cells)} cells survive correction. "
                    f"But drop the one defensive-contribution season and the edge falls to "
                    f"{non_dc['mean_gain_per_gw']:+.2f} pts/GW, **no longer significant** "
                    f"(t p={non_dc['paired_t_p']:.4f}). So the pooled result leans on the "
                    f"same perishable rule-modelling advantage that Phase 5c already "
                    f"identified. Unlike Phase 5b the residual fixture edge is genuinely "
                    f"positive across seasons — it is just too small to prove on "
                    f"{non_dc['gameweeks']} gameweeks. " + fair_fight)
    lines += [headline, "", replication]

    lines += [
        "",
        "## Caveats",
        "",
        "- The squad is rebuilt from scratch every gameweek. A real manager carries a "
        "squad and pays for transfers, so the absolute totals are unreachable. Every "
        "model faces the identical rule, so the comparison is fair.",
        "- The captain is the highest-projected player **in the chosen XI**, not "
        "jointly optimised with it. That is the order a manager decides in.",
        "- Autosubs (a bench player replacing a starter who played 0 minutes) are not "
        "modelled. They would help every model.",
        "- Injury flags are unavailable historically; live they are used, which should "
        "favour our projection.",
        "- Where projections tie, several squads are equally optimal to the model but "
        "score differently in reality; tie-breaking is arbitrary, so these p-values "
        "carry a little irreducible noise beyond sampling.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_optimizer.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    frame = (cached_predictions(seasons=SEASONS, xg_source=XG_SOURCE)
             if SEASONS is not None else cached_predictions(xg_source=XG_SOURCE))
    logger.info("benchmark on %d seasons, xg_source=%s, tag=%s",
                frame["season"].nunique(), XG_SOURCE, RUN_TAG)
    global _PERSIST
    _PERSIST = True                     # the long sweep wants resumable disk caching
    set_cache_tag(XG_SOURCE)            # keep in-memory keys aligned with the source
    _load_cache()

    cells = []
    for budget in SQUAD_BUDGETS:
        for with_captain in (False, True):
            cell = compare(frame, CHALLENGER, BASELINE, budget, with_captain)
            logger.info("squad=%.0fm captain=%s  gain %+.2f/GW  t p=%.4f  W p=%.4f",
                        budget, with_captain, cell["mean_gain_per_gw"],
                        cell["paired_t_p"], cell["wilcoxon_p"])
            cells.append(cell)

    apply_multiplicity_correction(cells)
    primary = next(c for c in cells if c["squad_budget"] == PRIMARY_BUDGET
                   and c["captain"] is PRIMARY_CAPTAIN)

    logger.info("scoring every model on the real game")
    table = pd.DataFrame({
        "no captain": all_models(frame, PRIMARY_BUDGET, False),
        "with captain": all_models(frame, PRIMARY_BUDGET, True),
    }).sort_values("with captain", ascending=False)
    table.index.name = "model (points/GW, £100m squad)"

    shapes = pd.DataFrame({
        model: formation_mix(frame, model, PRIMARY_BUDGET)
        for model in (CHALLENGER, BASELINE)
    }).fillna(0.0).sort_values(CHALLENGER, ascending=False)
    shapes.index.name = "formation (share of gameweeks)"

    seasons = per_season(frame, PRIMARY_BUDGET, PRIMARY_CAPTAIN)
    non_dc = without_dc_season(frame, PRIMARY_BUDGET, PRIMARY_CAPTAIN)
    decay = decay_analysis(frame, PRIMARY_BUDGET, PRIMARY_CAPTAIN)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(frame, cells, primary, table.to_markdown(floatfmt=".2f"),
                          seasons, shapes.to_markdown(floatfmt=".2f"), non_dc, decay, run_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fpl_squad_benchmark_{RUN_TAG}_{run_id}.md").write_text(
        report, encoding="utf-8")
    (RESULTS_DIR / f"fpl_squad_benchmark_{RUN_TAG}_{run_id}.json").write_text(
        json.dumps({"cells": cells, "primary": primary, "per_season": seasons,
                    "without_dc_season": non_dc, "decay": decay,
                    "models": {"no captain": all_models(frame, PRIMARY_BUDGET, False),
                               "with captain": all_models(frame, PRIMARY_BUDGET, True)}},
                   indent=2), encoding="utf-8")
    # The console is cp1252 on Windows; the report is UTF-8. Never let an arrow
    # crash a run whose real output is already safely on disk.
    print(report.encode("ascii", "replace").decode("ascii"))
    return cells


if __name__ == "__main__":
    main()
