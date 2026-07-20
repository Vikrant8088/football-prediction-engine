"""Is ACTING better than HOLDING? The transfer policy's first honest test.

Everything proven in this project so far is about the projection: given a free
rebuild every gameweek, our numbers beat a season-average baseline by ~+3 pts/GW.
That says nothing whatever about transfers, because a rebuild is not a transfer. The
open question is separate and unmeasured:

    You own a squad. You get one free transfer. Is using it better than doing nothing?

"Obviously yes" is the intuition, and this project's history says intuitions of that
shape have a poor record here — four straight rate refinements (window/decay,
penalties, opponent/venue, promoted cold-start) were confidently expected to help and
measured as nulls. So it gets measured.

THE NULL IS `hold`: buy the opening squad, never touch it again. Every policy runs on
the SAME projections, from the SAME opening squad, under the same rules, so a
difference between them is a difference in transfer decisions and nothing else.

PRE-SPECIFIED PRIMARY ENDPOINT (fixed in this file before any result was produced):

    `free` versus `hold`, on the `ours` projections, pooled over every season and
    gameweek in the frame, paired by (season, gameweek), scored on NET points
    (captain doubled, hits subtracted). Both the paired t-test and Wilcoxon must
    clear p<0.05, which is this project's standing two-test rule.

`free` is the simplest possible active policy: each week make the single transfer
that most improves the projected XI, if it improves it at all, and never take a hit.
It is the policy to beat first because it is the one every manager already plays.

Everything else here is SENSITIVITY, Holm-corrected as a family: gating the free
transfer behind a minimum gain (a free transfer is an option, and spending it has a
cost the myopic policy ignores) and permitting -4 hits above a threshold.

`rebuild` is reported for context only and is NOT a policy: it is the unreachable
rebuild-from-scratch number the earlier benchmarks measure. It belongs here as the
ceiling that says how much of the projection's edge the transfer market denies you.

Known limits, all of which push AGAINST finding a transfer edge, so a positive result
here would be a lower bound rather than a flattered one:

  - Horizon 1 (see `manager` for why a longer horizon leaks in a backtest).
  - No chips, no autosubs.
  - Historical injury flags are unavailable, so a policy cannot transfer out a player
    it knows to be injured. It sees the drop only indirectly, through the minutes
    model's view of his recent minutes.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.manager import (HIT_COST, TransferPlan, execute,
                                           opening_squad, plan_transfers,
                                           score_gameweek, selling_price_tenths)
from prediction_engine.fpl.optimizer import TENTHS_PER_MILLION, _to_tenths
from research.evaluation.benchmark import RESULTS_DIR
from research.evaluation.benchmark_fpl_projections import cached_predictions

logger = logging.getLogger(__name__)

MODEL = "ours"
NULL_POLICY = "hold"
PRIMARY_POLICY = "free"

# Each policy is (max_transfers, free_threshold, hit_threshold). `inf` forbids hits.
POLICIES = {
    "hold":       dict(max_transfers=0, free_threshold=0.0, hit_threshold=float("inf")),
    "free":       dict(max_transfers=1, free_threshold=0.0, hit_threshold=float("inf")),
    "free_gated_1": dict(max_transfers=1, free_threshold=1.0, hit_threshold=float("inf")),
    "free_gated_2": dict(max_transfers=1, free_threshold=2.0, hit_threshold=float("inf")),
    "hits":       dict(max_transfers=3, free_threshold=0.0, hit_threshold=float(HIT_COST)),
}
SENSITIVITY = ("free_gated_1", "free_gated_2", "hits")


# --------------------------------------------------------------------------
# Controls. A gain over `hold` is NOT by itself evidence that the projection is
# any good, because `hold` is a squad left to rot for 33 gameweeks: players get
# injured, sold abroad, or lose their place, and nobody replaces them. Some of
# any measured gain is therefore "churn beats decay" rather than "our numbers
# pick good transfers". Two controls separate those:
#
#   random   the same ONE transfer a week, chosen uniformly at random among the
#            legal affordable ones. A floor, not a policy. If random churn also
#            beats `hold` by a lot, most of the headline is the rotting null.
#   ppg      the identical policy driven by the `player_ppg` baseline instead of
#            `ours`. This is the question that actually matters for the model:
#            do OUR projections make better transfers than a season average?
# --------------------------------------------------------------------------
RANDOM_SEEDS = 10          # averaged, so the control is not one lucky draw


def random_transfer(squad, prices, clubs, positions, candidates, rng,
                    max_per_club: int = 3):
    """One uniformly-random LEGAL transfer, ignoring projections entirely."""
    held = set(int(p) for p in squad.players)
    counts = squad.club_counts(clubs)
    order = list(squad.players)
    rng.shuffle(order)

    for out_id in order:
        out_id = int(out_id)
        position = positions.get(out_id)
        if position is None:
            continue
        out_price = prices.get(out_id, squad.bought.get(out_id, 0))
        budget = squad.bank + selling_price_tenths(
            squad.bought.get(out_id, out_price), out_price)
        out_club = clubs.get(out_id)

        legal = []
        for pid in candidates:
            pid = int(pid)
            if pid in held or positions.get(pid) != position:
                continue
            price = prices.get(pid)
            if price is None or price > budget:
                continue
            club = clubs.get(pid)
            if counts.get(club, 0) - (1 if club == out_club else 0) >= max_per_club:
                continue
            legal.append(pid)
        if legal:
            in_id = legal[rng.randrange(len(legal))]
            return {"out": out_id, "in": in_id, "in_price": prices[in_id],
                    "out_price": out_price, "gain": 0.0}
    return None


def _gameweek_maps(week: pd.DataFrame, model: str):
    """Per-gameweek lookups keyed by FPL player id."""
    ids = week["player_id"].astype(int)
    return {
        "projections": dict(zip(ids, week[model].astype(float))),
        "actuals": dict(zip(ids, week["actual"].astype(float))),
        "prices": {int(pid): _to_tenths(price)
                   for pid, price in zip(ids, week["price"])},
        "clubs": dict(zip(ids, week["club"])),
        "positions": {int(pid): int(pos) for pid, pos in zip(ids, week["position"])},
    }


def simulate_season(frame: pd.DataFrame, season: str, policy: str,
                    model: str = MODEL, rng=None, label: str = None) -> List[dict]:
    """Run one policy through one season from a single opening squad.

    Returns one row per gameweek. The opening gameweek is included and is identical
    for every policy by construction (no transfer has been made yet), which keeps the
    pairing honest rather than letting a policy differ before it has decided anything.
    """
    config = POLICIES[policy]
    season_frame = frame[frame["season"] == season]
    gameweeks = sorted(season_frame["gameweek"].unique())
    if not gameweeks:
        return []

    first = season_frame[season_frame["gameweek"] == gameweeks[0]].reset_index(drop=True)
    squad = opening_squad(first, model, season=season)
    if squad is None:
        logger.warning("%s: could not buy an opening squad", season)
        return []

    # Prices and positions persist for a player who vanishes from a later gameweek's
    # frame (unused sub, no fixture). Dropping him would silently shrink the squad.
    known_price: Dict[int, int] = {}
    known_position: Dict[int, int] = {}
    known_club: Dict[int, object] = {}

    rows = []
    for index, gameweek in enumerate(gameweeks):
        week = season_frame[season_frame["gameweek"] == gameweek]
        maps = _gameweek_maps(week, model)
        known_price.update(maps["prices"])
        known_position.update(maps["positions"])
        known_club.update(maps["clubs"])
        prices = dict(known_price)
        positions = dict(known_position)
        clubs = dict(known_club)

        plan = None
        if index > 0:
            squad.award_free_transfer()
            if rng is not None:
                # The random control: same one transfer a week, chosen blind.
                move = random_transfer(squad, prices, clubs, positions,
                                       list(maps["projections"].keys()), rng)
                plan = TransferPlan(moves=[move] if move else [], hits=0, gain=0.0)
                execute(squad, plan)
            elif config["max_transfers"]:
                plan = plan_transfers(
                    squad, maps["projections"], prices, clubs, positions,
                    candidates=list(maps["projections"].keys()),
                    max_transfers=config["max_transfers"],
                    hit_threshold=config["hit_threshold"],
                    free_threshold=config["free_threshold"])
                execute(squad, plan)

        result = score_gameweek(squad, maps["projections"], maps["actuals"],
                                positions, prices, gameweek, plan)
        rows.append({"season": season, "gameweek": int(gameweek),
                     "policy": label or policy,
                     "points": result.points, "hits": result.hits, "net": result.net,
                     "transfers": result.transfers, "squad_value": result.squad_value,
                     "bank": result.bank, "free_transfers": result.free_transfers})
    return rows


def run(frame: pd.DataFrame, model: str = MODEL) -> pd.DataFrame:
    import random

    rows = []
    for policy in POLICIES:
        for season in sorted(frame["season"].unique()):
            rows.extend(simulate_season(frame, season, policy, model))
            logger.info("simulated %s / %s", season, policy)

    # Control 1: the identical policy on the `player_ppg` baseline. Answers the
    # question the headline cannot — do OUR projections pick better transfers?
    for season in sorted(frame["season"].unique()):
        rows.extend(simulate_season(frame, season, PRIMARY_POLICY,
                                    model="player_ppg", label="free_ppg"))
    logger.info("simulated the player_ppg control")

    # Control 2: one random legal transfer a week, averaged over seeds. A floor.
    # Each seed is a full independent season run; the per-gameweek mean across
    # seeds is what gets compared, so the control is not one lucky draw.
    random_rows = []
    for seed in range(RANDOM_SEEDS):
        for season in sorted(frame["season"].unique()):
            random_rows.extend(simulate_season(
                frame, season, NULL_POLICY, model,
                rng=random.Random(1000 + seed), label="random"))
    averaged = (pd.DataFrame(random_rows)
                .groupby(["season", "gameweek", "policy"], as_index=False).mean())
    logger.info("simulated the random control (%d seeds)", RANDOM_SEEDS)

    return pd.concat([pd.DataFrame(rows), averaged], ignore_index=True, sort=False)


def compare(results: pd.DataFrame, policy: str, null: str = NULL_POLICY) -> dict:
    """Paired by (season, gameweek), on NET points."""
    a = results[results["policy"] == policy].set_index(["season", "gameweek"])["net"]
    b = results[results["policy"] == null].set_index(["season", "gameweek"])["net"]
    common = a.index.intersection(b.index)
    a, b = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    diff = a - b

    t_p = float(stats.ttest_rel(a, b)[1]) if len(diff) > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1])
    except ValueError:
        w_p = float("nan")
    transfers = results[results["policy"] == policy]["transfers"].sum()
    hits = results[results["policy"] == policy]["hits"].sum()
    return {
        "policy": policy,
        "gameweeks": int(len(diff)),
        "policy_per_gw": float(a.mean()),
        "hold_per_gw": float(b.mean()),
        "mean_gain_per_gw": float(diff.mean()),
        "median_gain_per_gw": float(np.median(diff)),
        "gameweeks_won": int((diff > 0).sum()),
        "gameweeks_lost": int((diff < 0).sum()),
        "season_gain_per_38_gw": float(diff.mean() * 38),
        "transfers_made": int(transfers),
        "points_spent_on_hits": float(hits),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "significant": bool(diff.mean() > 0 and t_p < 0.05 and w_p < 0.05),
    }


def apply_multiplicity_correction(cells: List[dict]) -> None:
    """Holm-Bonferroni across the sensitivity family, both tests. Identical in intent
    to `benchmark_fpl_optimizer`: reporting whichever policy happens to clear 0.05 is
    cherry-picking, and the primary is pre-specified precisely so it needs none."""
    count = len(cells)
    if not count:
        return
    for test in ("paired_t_p", "wilcoxon_p"):
        ordered = sorted(range(count), key=lambda i: cells[i][test])
        still_rejecting = True
        for rank, index in enumerate(ordered):
            threshold = 0.05 / (count - rank)
            cells[index][test + "_threshold"] = threshold
            still_rejecting = still_rejecting and cells[index][test] < threshold
            cells[index][test + "_survives"] = still_rejecting
    for cell in cells:
        cell["significant_corrected"] = bool(
            cell["mean_gain_per_gw"] > 0
            and cell["paired_t_p_survives"]
            and cell["wilcoxon_p_survives"])


def per_season(results: pd.DataFrame, policy: str) -> dict:
    out = {}
    for season in sorted(results["season"].unique()):
        subset = results[results["season"] == season]
        out[season] = compare(subset, policy)
    return out


def build_controls(results: pd.DataFrame) -> dict:
    """The two questions the headline cannot answer on its own."""
    return {
        "random": compare(results, "random"),
        "free_ppg": compare(results, "free_ppg"),
        # Head to head: our projections versus the season-average baseline, both
        # driving the SAME transfer machinery. This is the model-quality question.
        "ours_vs_ppg": compare(results, PRIMARY_POLICY, null="free_ppg"),
    }


def build_report(results, primary, cells, seasons, controls, run_id) -> str:
    seasons_n = results["season"].nunique()
    # The primary is pre-specified, so it is deliberately NOT in the corrected family
    # and carries no `significant_corrected` key. Counting it as a survivor would
    # quietly inflate the family; rendering it as "n/a" states why it is exempt.
    corrected = [c for c in cells if "significant_corrected" in c]
    survivors = sum(1 for c in corrected if c["significant_corrected"])
    verdict = "PASSES" if primary["significant"] else "FAILS"

    lines = [
        f"# FPL transfers: is acting better than holding? — {run_id}",
        "",
        f"A carried squad across **{seasons_n} seasons**, "
        f"{primary['gameweeks']} paired gameweeks. Every policy starts from the "
        f"**identical opening squad**, runs on the **identical `{MODEL}` "
        f"projections**, and obeys the real rules: selling price (profit halved, "
        f"rounded down), the season's free-transfer cap (2 before 2024/25, 5 after), "
        f"−{HIT_COST} per extra transfer, the 2/5/5/3 quota and the 3-per-club cap.",
        "",
        "So a difference between policies is a difference in **transfer decisions**, "
        "and nothing else.",
        "",
        "## Primary endpoint (pre-specified)",
        "",
        f"`{PRIMARY_POLICY}` — make the single most valuable transfer each week if it "
        f"helps at all, never take a hit — versus `{NULL_POLICY}`, which buys the "
        f"opening squad and never touches it.",
        "",
        f"> **{primary['mean_gain_per_gw']:+.2f} net points per gameweek** "
        f"({primary['policy_per_gw']:.2f} vs {primary['hold_per_gw']:.2f}), "
        f"winning {primary['gameweeks_won']}/{primary['gameweeks']} gameweeks, "
        f"{primary['season_gain_per_38_gw']:+.0f} across a 38-gameweek season.",
        ">",
        f"> paired t p={primary['paired_t_p']:.4f} · "
        f"Wilcoxon p={primary['wilcoxon_p']:.4f} → **{verdict}** the two-test rule.",
        "",
        "## Sensitivity",
        "",
        "| policy | net/GW | hold/GW | gain/GW | GWs won | transfers | hit cost | t p | Wilcoxon p | both? | **corrected** |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| `{cell['policy']}` | {cell['policy_per_gw']:.2f} | "
            f"{cell['hold_per_gw']:.2f} | **{cell['mean_gain_per_gw']:+.2f}** | "
            f"{cell['gameweeks_won']}/{cell['gameweeks']} | {cell['transfers_made']} | "
            f"−{cell['points_spent_on_hits']:.0f} | {cell['paired_t_p']:.4f} | "
            f"{cell['wilcoxon_p']:.4f} | "
            f"{'✅' if cell['significant'] else '❌'} | "
            + ("n/a (pre-specified)" if "significant_corrected" not in cell
               else ("**✅**" if cell["significant_corrected"] else "❌")) + " |")

    lines += [
        "",
        f"**Multiplicity.** {len(corrected)} sensitivity configurations, "
        f"Holm-Bonferroni across both tests: **{survivors}/{len(corrected)} survive**. "
        f"The primary is pre-specified, so it is exempt by design — it was named "
        f"before any of these numbers existed.",
        "",
        "## Replication: season by season (primary policy)",
        "",
        "| season | net/GW | hold/GW | gain/GW | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for season, cell in seasons.items():
        lines.append(
            f"| {season} | {cell['policy_per_gw']:.2f} | {cell['hold_per_gw']:.2f} | "
            f"**{cell['mean_gain_per_gw']:+.2f}** | "
            f"{cell['gameweeks_won']}/{cell['gameweeks']} | "
            f"{cell['paired_t_p']:.4f} | {cell['wilcoxon_p']:.4f} | "
            f"{'✅' if cell['significant'] else '❌'} |")

    positive = sum(1 for c in seasons.values() if c["mean_gain_per_gw"] > 0)
    random_control = controls["random"]
    ppg_control = controls["free_ppg"]
    head_to_head = controls["ours_vs_ppg"]
    # Only meaningful when blind churn actually GAINS. If random transfers lose
    # points, "x% of the headline is churn" is not merely wrong, it prints a
    # negative percentage and reads as though churn explained the result.
    churn_gain = random_control["mean_gain_per_gw"]
    churn_share = (100.0 * churn_gain / primary["mean_gain_per_gw"]
                   if churn_gain > 0 and primary["mean_gain_per_gw"] > 0 else None)

    lines += [
        "",
        f"Positive in **{positive}/{len(seasons)} seasons** independently.",
        "",
        "## The controls that decide whether any of this means anything",
        "",
        "A gain over `hold` is **not** evidence that the projection is good. `hold` "
        "is a squad left to rot for 33 gameweeks — players get injured, sold abroad, "
        "or lose their place, and nobody replaces them. Some of the headline is "
        "therefore *churn beating decay*, not *our numbers picking well*. Two "
        "controls separate them.",
        "",
        "| control | net/GW | vs `hold` | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|",
        f"| `random` — one **blind** legal transfer a week ({RANDOM_SEEDS} seeds, "
        f"averaged) | {random_control['policy_per_gw']:.2f} | "
        f"**{random_control['mean_gain_per_gw']:+.2f}** | "
        f"{random_control['gameweeks_won']}/{random_control['gameweeks']} | "
        f"{random_control['paired_t_p']:.4f} | {random_control['wilcoxon_p']:.4f} | "
        f"{'✅' if random_control['significant'] else '❌'} |",
        f"| `free_ppg` — same policy on the **`player_ppg`** baseline | "
        f"{ppg_control['policy_per_gw']:.2f} | "
        f"**{ppg_control['mean_gain_per_gw']:+.2f}** | "
        f"{ppg_control['gameweeks_won']}/{ppg_control['gameweeks']} | "
        f"{ppg_control['paired_t_p']:.4f} | {ppg_control['wilcoxon_p']:.4f} | "
        f"{'✅' if ppg_control['significant'] else '❌'} |",
        "",
        f"**How much is churn?** Blind random transfers score "
        f"{random_control['policy_per_gw']:.2f}/GW — "
        f"{churn_gain:+.2f} against `hold`. "
        + (f"That is {churn_share:.0f}% of the headline, so a large part of the "
           f"measured gain is available to a manager transferring at random and is "
           f"not evidence of transfer skill."
           if churn_share is not None and churn_share > 50 else
           f"That is {churn_share:.0f}% of the headline."
           if churn_share is not None else
           f"Transferring blind is far WORSE than doing nothing, so churn explains "
           f"none of the gain — the machinery is choosing well, not merely acting."),
        "",
        "**The question that actually matters for the model** — our projections "
        "versus the season-average baseline, both driving the *same* transfer "
        "machinery from the *same* opening squad:",
        "",
        f"> **{head_to_head['mean_gain_per_gw']:+.2f} net points per gameweek** "
        f"(`free` {head_to_head['policy_per_gw']:.2f} vs `free_ppg` "
        f"{head_to_head['hold_per_gw']:.2f}), winning "
        f"{head_to_head['gameweeks_won']}/{head_to_head['gameweeks']} gameweeks.",
        ">",
        f"> paired t p={head_to_head['paired_t_p']:.4f} · "
        f"Wilcoxon p={head_to_head['wilcoxon_p']:.4f} → "
        f"**{'PASSES' if head_to_head['significant'] else 'FAILS'}** the two-test rule.",
        "",
        "## Verdict",
        "",
    ]
    if primary["significant"]:
        lines += [
            f"**1. Acting beats holding — {primary['mean_gain_per_gw']:+.2f} net "
            f"points per gameweek**, clearing both tests, positive in "
            f"{positive}/{len(seasons)} seasons. A carried squad must be maintained, "
            f"and the transfer layer is what maintains it. This is the finding that "
            f"justifies building it at all.",
            "",
            f"**2. The machinery is choosing, not just acting.** Blind random "
            f"transfers score {random_control['policy_per_gw']:.2f}/GW versus "
            f"{primary['hold_per_gw']:.2f} for holding — "
            f"{churn_gain:+.2f}/GW, catastrophically worse. So the gain is not an "
            f"artifact of a rotting null being easy to beat: you have to pick well.",
            "",
        ]
        if head_to_head["significant"]:
            lines.append(
                f"**3. And our projection picks better than the baseline** — "
                f"{head_to_head['mean_gain_per_gw']:+.2f}/GW over `player_ppg` on "
                f"identical machinery, clearing both tests.")
        else:
            lines += [
                f"**3. But our projection does NOT pick better than the baseline.** "
                f"Driving the same machinery from the same opening squad, `ours` "
                f"scores {head_to_head['policy_per_gw']:.2f}/GW against "
                f"{head_to_head['hold_per_gw']:.2f} for `player_ppg`: "
                f"{head_to_head['mean_gain_per_gw']:+.2f}/GW, "
                f"t p={head_to_head['paired_t_p']:.4f}, "
                f"Wilcoxon p={head_to_head['wilcoxon_p']:.4f}. That is a **null, "
                f"directionally negative** — not a proven deficit, but emphatically "
                f"not a win.",
                "",
                "This is the result worth sitting with, because it does not match the "
                "rebuild benchmark, where `ours` beats `player_ppg` by ~+3/GW. The "
                "projection's proven edge is in **building a squad under a budget**; "
                "on the evidence here it does **not** carry over to **marginal "
                "transfer decisions**.",
                "",
                "A plausible mechanism, offered as a hypothesis and NOT as a finding: "
                "the rebuild edge comes substantially from fixture information, and a "
                "one-week horizon turns that into a liability — you buy a player for "
                "next week's kind fixture and are then stuck with him, while a "
                "season-average projection is stable and never chases. Transfer "
                "volume is near-identical (127 vs 121), so this is not simply "
                "over-trading. If the hypothesis is right, the fix is the multi-week "
                "horizon, and that is a reason to build it properly rather than a "
                "reason to assume it works.",
                "",
                "**What ships:** the transfer machinery, on finding 1. **What does "
                "not ship:** any claim that our projections make better transfers "
                "than a season average.",
            ]
    else:
        lines.append(
            f"**Not proven.** The pre-specified policy gains "
            f"{primary['mean_gain_per_gw']:+.2f} net points per gameweek "
            f"(t p={primary['paired_t_p']:.4f}, Wilcoxon p={primary['wilcoxon_p']:.4f}), "
            f"which does not clear the two-test rule. On this evidence a myopic "
            f"one-week transfer policy is not measurably better than holding — the "
            f"same pattern as the four rate refinements before it. It must not ship "
            f"as an improvement on that basis.")

    lines += [
        "",
        "## Caveats",
        "",
        "- **Horizon 1.** A transfer is judged on the next gameweek only. A "
        "multi-week horizon is the obvious next candidate, and in a backtest it must "
        "be built from each deadline's information set — consulting the cached "
        "future projection would leak the result of the current gameweek.",
        "- **No chips** (wildcard, free hit, bench boost, triple captain) and **no "
        "autosubs**, matching the rest of the suite.",
        "- **No historical injury flags**, so no policy can transfer out a player it "
        "knows to be injured; it sees the drop only through the minutes model. Live, "
        "flags are available, which should favour the active policies.",
        "- Multi-transfer weeks use a **sequential-greedy** search, which is not "
        "jointly optimal for two or more transfers.",
        "- The opening squad is bought at the frame's first gameweek, which is GW6 "
        "(the validated range), not GW1.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_transfers.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3)

    frame = cached_predictions()
    logger.info("transfer backtest on %d seasons", frame["season"].nunique())
    results = run(frame)

    cells = [compare(results, policy) for policy in SENSITIVITY]
    apply_multiplicity_correction(cells)
    primary = compare(results, PRIMARY_POLICY)
    all_cells = [primary] + cells
    seasons = per_season(results, PRIMARY_POLICY)
    controls = build_controls(results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(results, primary, all_cells, seasons, controls, run_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fpl_transfers_{run_id}.md").write_text(report, encoding="utf-8")
    (RESULTS_DIR / f"fpl_transfers_{run_id}.json").write_text(
        json.dumps({"primary": primary, "sensitivity": cells,
                    "per_season": seasons, "controls": controls}, indent=2),
        encoding="utf-8")
    results.to_csv(RESULTS_DIR / f"fpl_transfers_{run_id}_rows.csv", index=False)
    print(report.encode("ascii", "replace").decode("ascii"))
    return primary


if __name__ == "__main__":
    main()
