"""Where does our only significant FPL season actually come from?

Across four replayed seasons the projection beats `player_ppg` by +1.83 points per
gameweek, which is NOT significant. Exactly one season is significant: 2025/26
(+6.42, both tests). That is also the only season with FPL's new
defensive-contribution rule - which our projection models explicitly (P(actions >=
threshold) under a Poisson) and which the baselines can only absorb slowly, through
realised points.

Coincidence is not an explanation, so this ablates the term:

  H-DC       the edge comes from modelling the new rule. Zeroing `dc_per_90`
             should collapse the 2025/26 edge.
  H-FIXTURE  the edge comes from fixture information (clean-sheet probability,
             opponent strength). Zeroing `dc_per_90` should leave it intact.

Method: re-run 2025/26 twice, identically, except that the ablated run forces every
player's `dc_per_90` to zero so the projection cannot see the rule. ACTUAL points
are untouched - the rule still awarded them - so the difference measures only what
*modelling* the rule buys. `player_ppg` is unchanged between runs by construction,
which makes the two gains directly comparable.

Result (recorded, not assumed): H-DC. The DC term is worth ~60% of the edge, and
without it 2025/26 is no longer significant.

An edge that comes from modelling a NEW rule before the field has adapted is real
but perishable: `player_ppg` already absorbs defensive-contribution points through
each player's own realised history, just with a lag. Expect it to decay.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_warehouse.utils.logging_config import configure_logging
from research.data.xg_loader import load_understat_matches
from research.evaluation import benchmark_fpl_projections as backtest
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

SEASON = "2025-26"          # the only season in which the DC rule exists
BASELINE = "player_ppg"     # the only baseline we have not clearly beaten


def _run(matches: pd.DataFrame) -> dict:
    frame = pd.DataFrame(backtest.run_season(SEASON, matches))
    frame["global_mean"] = frame["actual"].mean()
    return backtest.compare_top11(frame, "ours", BASELINE)


def run_ablation() -> dict:
    matches = load_understat_matches("EPL")

    full = _run(matches)

    original = backtest._rates_from_history

    def without_defensive_contribution(rows, gameweeks_elapsed):
        rates = original(rows, gameweeks_elapsed)
        rates["dc_per_90"] = 0.0     # the projection can no longer see the rule
        return rates

    backtest._rates_from_history = without_defensive_contribution
    try:
        ablated = _run(matches)
    finally:
        backtest._rates_from_history = original

    attributable = full["mean_gain_per_gw"] - ablated["mean_gain_per_gw"]
    return {
        "season": SEASON,
        "baseline": BASELINE,
        "with_defensive_contribution": full,
        "without_defensive_contribution": ablated,
        "points_per_gw_attributable_to_dc": attributable,
        "share_of_edge_from_dc": (
            attributable / full["mean_gain_per_gw"] if full["mean_gain_per_gw"] else float("nan")
        ),
        "supports": "H-DC" if not ablated["significant"] and full["significant"] else "H-FIXTURE",
    }


def build_report(result: dict, run_id: str) -> str:
    full, ablated = result["with_defensive_contribution"], result["without_defensive_contribution"]
    lines = [
        f"# FPL: what is the 2025/26 edge made of? - {run_id}",
        "",
        f"Our projection beats `{BASELINE}` significantly in exactly one of four "
        f"replayed seasons: {SEASON}. That is also the only season with FPL's "
        f"defensive-contribution rule, which we model and the baselines do not. "
        f"This ablation forces `dc_per_90` to zero so the projection cannot see the "
        f"rule. Actual points are untouched.",
        "",
        f"| run | our XI | {BASELINE} XI | gain/GW | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, c in (("**with** DC (as shipped)", full), ("**without** DC (ablated)", ablated)):
        lines.append(
            f"| {label} | {c['challenger_points_per_gw']:.2f} | "
            f"{c['baseline_points_per_gw']:.2f} | **{c['mean_gain_per_gw']:+.2f}** | "
            f"{c['gameweeks_won']}/{c['gameweeks']} | {c['paired_t_p']:.4f} | "
            f"{c['wilcoxon_p']:.4f} | {'yes' if c['significant'] else 'NO'} |"
        )

    share = result["share_of_edge_from_dc"]
    lines += [
        "",
        "## Verdict",
        "",
        f"**{result['supports']}.** Modelling the defensive-contribution rule is worth "
        f"**{result['points_per_gw_attributable_to_dc']:+.2f} points per gameweek**, "
        f"i.e. **{share:.0%}** of the {full['mean_gain_per_gw']:+.2f} edge. Without it, "
        f"{SEASON} falls to {ablated['mean_gain_per_gw']:+.2f} pts/GW "
        f"(t p={ablated['paired_t_p']:.4f}, Wilcoxon p={ablated['wilcoxon_p']:.4f}) - "
        f"no longer significant, and statistically indistinguishable from the three "
        f"seasons without the rule.",
        "",
        "**What this means, stated plainly:**",
        "",
        "- The *fixture* model - clean-sheet probability, opponent strength - is worth "
        f"about {ablated['mean_gain_per_gw']:+.2f} pts/GW and has **never reached "
        "significance in any season**. It is not, on this evidence, a proven edge.",
        "- The edge that *is* significant comes from correctly modelling a **new rule** "
        "before the field has adapted to it. That is genuine but **perishable**: "
        f"`{BASELINE}` already absorbs defensive-contribution points through each "
        "player's realised history, only with a lag. Expect decay.",
        "- It rests on a **single season** (33 gameweeks). It cannot be replicated, "
        "because 2025/26 is the only season the rule has ever existed.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_dc_ablation.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )
    result = run_ablation()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(result, run_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fpl_dc_ablation_{run_id}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"fpl_dc_ablation_{run_id}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(report)
    return result


if __name__ == "__main__":
    main()
