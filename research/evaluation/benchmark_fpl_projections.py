"""Do our FPL projections actually beat the obvious alternatives?

The scoring rules are proven exact (2,085/2,085 real matches). That says nothing
about whether the PROJECTIONS are any good. This is the test that decides whether
this is a product or a toy, and it is deliberately unkind to us.

Walk-forward by gameweek, replayed independently across every season in which
FPL publishes xG (2022/23 onward): to project gameweek k, a model may use only
gameweeks 1..k-1 of that season (for player rates) and only matches played before
that gameweek's first kickoff (for the team model). Nothing else.

The competitors are the things a real FPL manager would actually do instead:

  global_mean   predict the same number for everyone (the floor)
  player_ppg    the player's own points-per-gameweek so far     <- the real bar
  player_form5  the player's mean over his last five gameweeks
  price         rank by cost - "just pick the expensive ones"
  ours          the fixture-aware projection

Metrics, in ascending order of what an FPL manager cares about:
  MAE / RMSE      how close the number is
  Spearman rho    whether the RANKING is right (you pick players, not numbers)
  legal XI points field the best XI this model could ACTUALLY field - a legal
                  formation, max 3 per club - and sum what those players really
                  scored. This is the decision the manager makes, so it decides.

Four seasons give 131 paired gameweeks rather than 33. Just as importantly they
give FOUR INDEPENDENT REPLICATIONS: an edge that survives in every season is a
different kind of claim from one that appears once. Both are reported.

Two handicaps and three corrections, all stated up front:

  HANDICAP  FPL's injury flags are published only for the CURRENT moment, not
            historically, so this backtest cannot use them. Live, the projection
            does. The backtest therefore *understates* the real system - it is
            forced to project points for players who were injured and did not play.

  HANDICAP  Seasons before 2025/26 have no `defensive_contribution` column,
            because the rule did not exist. Those seasons neither award nor
            project those points. Consistent within each season.

  FIX       The XI is now LEGAL. The previous metric - rank by projected points,
            take the top 11 - is invalid: it fielded 5.9 goalkeepers on average
            in 2022/23, and a manager may field one. Clean-sheet and save points
            make keepers look efficient in isolation, so the naive metric
            punished whichever model rated them best, which was ours. Both the
            legal and the naive number are reported, because this defect is the
            reason the one-season Phase 5b result was wrong.

  FIX       Double gameweeks: a player with two fixtures in one gameweek is now a
            SINGLE row whose projection and actual points are summed. Previously
            he appeared twice and could be picked twice in one XI.

  FIX       `price` is now the player's price AT THAT GAMEWEEK, not his
            end-of-season price. End-of-season price is contaminated by the
            season's own results (good players rise), which flattered the
            baseline we are trying to beat.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_warehouse.utils.logging_config import configure_logging
from prediction_engine.fpl.projection import fixture_context, project_player, team_scoring_rates
from prediction_engine.scoreline_ensemble import ScorelineEnsemble
from research.data.fpl_archive import SEASONS_WITH_XG, load_gameweeks
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import RESULTS_DIR

logger = logging.getLogger(__name__)

FIRST_SCORED_GAMEWEEK = 6      # need some history before any model can say anything
MIN_TRAINING_SEASONS = 3       # the team model's own requirement
SQUAD_SIZE = 11                # a starting XI
FORM_WINDOW = 5
MODELS = ["ours", "player_ppg", "player_form5", "price", "global_mean"]

GKP, DEF, MID, FWD = 1, 2, 3, 4
MAX_PER_CLUB = 3               # FPL's squad rule
# Every legal FPL formation: exactly 1 keeper, then 3-5 DEF, 2-5 MID, 1-3 FWD.
FORMATIONS = [
    (defenders, midfielders, forwards)
    for defenders in range(3, 6)
    for midfielders in range(2, 6)
    for forwards in range(1, 4)
    if defenders + midfielders + forwards == SQUAD_SIZE - 1
]

# A manager never picks from the ~700 fringe players who score 0-2. The pool is
# defined using ONLY prior information (points-per-gameweek so far), so this is a
# legitimate prediction-time filter, not hindsight.
PICKABLE_PPG = 3.0

RATE_FIELDS = ("xg_per_90", "xa_per_90", "saves_per_90", "bonus_per_90",
               "dc_per_90", "cards_per_90")


def _rates_from_history(rows: list, gameweeks_elapsed: int) -> dict:
    """Per-90 rates from a player's completed fixtures only.

    `gameweeks_elapsed` counts gameweeks that actually took place (2022/23 GW7
    was cancelled), not `gameweek - 1`.
    """
    minutes = sum(r["minutes"] for r in rows)
    per_90 = minutes / 90.0 if minutes > 0 else 0.0

    def total(field):
        return sum(r[field] for r in rows)

    return {
        "minutes": minutes,
        "gameweeks": max(gameweeks_elapsed, 1),
        "xg_per_90": (total("expected_goals") / per_90) if per_90 else 0.0,
        "xa_per_90": (total("expected_assists") / per_90) if per_90 else 0.0,
        "saves_per_90": (total("saves") / per_90) if per_90 else 0.0,
        "bonus_per_90": (total("bonus") / per_90) if per_90 else 0.0,
        "dc_per_90": (total("defensive_contribution") / per_90) if per_90 else 0.0,
        "cards_per_90": ((total("yellow_cards") + 3 * total("red_cards")) / per_90) if per_90 else 0.0,
    }


def _project_gameweek(model, rates, fixtures, history_rates, position):
    """Sum a player's projection over every fixture he plays this gameweek."""
    total = 0.0
    for fixture in fixtures:
        team, opponent = fixture["team"], fixture["opponent"]
        if team not in rates or opponent not in rates:
            return None
        home, away = (team, opponent) if fixture["was_home"] else (opponent, team)
        grid = model.grid(home, away)
        if grid is None:
            return None
        context = model.context(home, away, is_home=fixture["was_home"])

        projection_input = pd.Series({
            "position": position,
            "minutes": history_rates["minutes"],
            # Historical availability is not published; live it is used.
            "available": True,
            "chance_of_playing": 100.0,
            **{field: history_rates[field] for field in RATE_FIELDS},
        })
        total += project_player(
            projection_input, context, rates[team], gameweeks=history_rates["gameweeks"]
        )["expected_points"]
    return total


class _GridCache:
    """One fitted team model per gameweek; grids and contexts memoised per fixture."""

    def __init__(self, fitted):
        self._fitted = fitted
        self._grids = {}
        self._contexts = {}

    def grid(self, home, away):
        if (home, away) not in self._grids:
            try:
                self._grids[(home, away)] = self._fitted.scoreline_grid(home, away)
            except Exception:
                self._grids[(home, away)] = None
        return self._grids[(home, away)]

    def context(self, home, away, is_home):
        key = (home, away, is_home)
        if key not in self._contexts:
            self._contexts[key] = fixture_context(self.grid(home, away), is_home=is_home)
        return self._contexts[key]


def run_season(season: str, matches: pd.DataFrame) -> list:
    """Walk forward through one season, gameweek by gameweek."""
    frame = load_gameweeks(season)

    fixtures_by_gw = defaultdict(lambda: defaultdict(list))   # gw -> player -> [fixture]
    for row in frame.to_dict("records"):
        fixtures_by_gw[row["gameweek"]][row["player_id"]].append(row)

    positions = frame.groupby("player_id")["position"].first().to_dict()
    names = frame.groupby("player_id")["player"].first().to_dict()

    history = defaultdict(list)        # player -> fixture rows so far
    gw_totals = defaultdict(list)      # player -> per-gameweek points so far
    predictions = []

    for elapsed, gameweek in enumerate(sorted(fixtures_by_gw)):
        by_player = fixtures_by_gw[gameweek]

        if gameweek >= FIRST_SCORED_GAMEWEEK:
            kickoff = min(f["kickoff_time"] for fs in by_player.values() for f in fs)
            cutoff = pd.Timestamp(kickoff).tz_convert(None)
            train = matches[matches["date"] < cutoff]

            if train["season"].nunique() >= MIN_TRAINING_SEASONS:
                model = _GridCache(ScorelineEnsemble().fit(train))
                rates = team_scoring_rates(train)
                logger.info("%s GW%-2d  training on %d matches", season, gameweek, len(train))

                for player_id, fixtures in by_player.items():
                    prior_totals = gw_totals[player_id]
                    if not prior_totals:
                        continue
                    history_rates = _rates_from_history(history[player_id], elapsed)
                    projected = _project_gameweek(
                        model, rates, fixtures, history_rates, positions[player_id]
                    )
                    if projected is None:
                        continue
                    predictions.append({
                        "season": season,
                        "gameweek": gameweek,
                        "player_id": player_id,
                        "player": names[player_id],
                        "position": positions[player_id],
                        "club": fixtures[0]["team"],   # for the max-3-per-club rule
                        "actual": sum(f["total_points"] for f in fixtures),
                        "ours": projected,
                        "player_ppg": float(np.mean(prior_totals)),
                        "player_form5": float(np.mean(prior_totals[-FORM_WINDOW:])),
                        # Price AT this gameweek - no end-of-season contamination.
                        "price": float(fixtures[0]["price"]),
                    })

        for player_id, fixtures in by_player.items():
            history[player_id].extend(fixtures)
            gw_totals[player_id].append(sum(f["total_points"] for f in fixtures))

    return predictions


def run_backtest(seasons=SEASONS_WITH_XG) -> pd.DataFrame:
    matches = load_understat_matches("EPL")
    predictions = []
    for season in seasons:
        predictions.extend(run_season(season, matches))

    frame = pd.DataFrame(predictions)
    frame["global_mean"] = frame["actual"].mean()   # constant floor
    return frame


def _unconstrained_top11(week: pd.DataFrame, model: str) -> float:
    """The naive 'best 11 by projected points'. Kept ONLY to document why it is
    invalid: it will happily field six goalkeepers, and did."""
    if len(week) < SQUAD_SIZE:
        return np.nan
    return float(week.nlargest(SQUAD_SIZE, model)["actual"].sum())


def _best_legal_xi(week: pd.DataFrame, model: str) -> float:
    """Actual points scored by the best XI this model could ACTUALLY field.

    An FPL manager may not pick eleven goalkeepers. Ranking players by projected
    points and taking the top 11 does exactly that - the unconstrained metric
    used before this fix filled 5.9 of our 11 slots with keepers in 2022/23,
    because clean-sheet and save points make them look efficient in isolation.
    That is a defect in the YARDSTICK, not in the projection, and it silently
    penalises whichever model likes keepers most.

    So: enforce a legal formation (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD) and FPL's
    max-3-players-per-club rule. Within each formation, players are taken in
    descending projected points, skipping any whose club is already full; the
    formation with the highest projected total is fielded, and we then score what
    those players REALLY got.

    Greedy fill (rather than an exact solve) can occasionally leave a fractionally
    better legal XI on the table, but it is applied identically to every model, so
    the comparison stays fair. Budget is still ignored - see the report's caveats.
    """
    squad = select_legal_xi(week, model)
    return np.nan if squad is None else float(squad["actual"].sum())


def select_legal_xi(week: pd.DataFrame, model: str):
    """The eleven players this model would field. `None` if no legal XI exists."""
    if len(week) < SQUAD_SIZE:
        return None

    ordered = week.sort_values(model, ascending=False)
    by_position = {position: ordered[ordered["position"] == position]
                   for position in (GKP, DEF, MID, FWD)}

    best_projected, best_squad = None, None
    for defenders, midfielders, forwards in FORMATIONS:
        needed = {GKP: 1, DEF: defenders, MID: midfielders, FWD: forwards}
        clubs = defaultdict(int)
        squad, feasible = [], True

        for position in (GKP, DEF, MID, FWD):
            picked = []
            for index, player in by_position[position].iterrows():
                if len(picked) == needed[position]:
                    break
                if clubs[player["club"]] >= MAX_PER_CLUB:
                    continue
                clubs[player["club"]] += 1
                picked.append(index)
            if len(picked) < needed[position]:
                feasible = False       # blank gameweek: not enough of this position
                break
            squad.extend(picked)

        if not feasible:
            continue
        projected = float(week.loc[squad, model].sum())
        if best_projected is None or projected > best_projected:
            best_projected, best_squad = projected, squad

    return None if best_squad is None else week.loc[best_squad]


def _top11(frame: pd.DataFrame, model: str, selector=_best_legal_xi) -> pd.Series:
    """Actual points scored by each model's XI, per season-gameweek."""
    return frame.groupby(["season", "gameweek"]).apply(
        lambda week: selector(week, model)
    ).dropna()


def evaluate(frame: pd.DataFrame) -> dict:
    pickable = frame[frame["player_ppg"] >= PICKABLE_PPG]

    results = {}
    for model in MODELS:
        errors = frame[model] - frame["actual"]
        spearman, spearman_pool = [], []
        for key, week in frame.groupby(["season", "gameweek"]):
            if len(week) >= SQUAD_SIZE:
                rho = stats.spearmanr(week[model], week["actual"]).correlation
                if not np.isnan(rho):
                    spearman.append(rho)
        for key, pool in pickable.groupby(["season", "gameweek"]):
            if len(pool) >= SQUAD_SIZE:
                rho = stats.spearmanr(pool[model], pool["actual"]).correlation
                if not np.isnan(rho):
                    spearman_pool.append(rho)

        legal = _top11(frame, model)
        naive = _top11(frame, model, selector=_unconstrained_top11)
        keepers = _mean_keepers_in_naive_xi(frame, model)
        results[model] = {
            "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())),
            "spearman_all": float(np.mean(spearman)) if spearman else float("nan"),
            "spearman_pickable": float(np.mean(spearman_pool)) if spearman_pool else float("nan"),
            "legal_xi_points_per_gw": float(legal.mean()) if len(legal) else float("nan"),
            "naive_top11_points_per_gw": float(naive.mean()) if len(naive) else float("nan"),
            "naive_xi_goalkeepers": keepers,
        }
    return results


def _mean_keepers_in_naive_xi(frame: pd.DataFrame, model: str) -> float:
    """How many goalkeepers the invalid metric would have fielded. A manager gets 1."""
    counts = frame.groupby(["season", "gameweek"]).apply(
        lambda week: (week.nlargest(SQUAD_SIZE, model)["position"] == GKP).sum()
        if len(week) >= SQUAD_SIZE else np.nan
    ).dropna()
    return float(counts.mean()) if len(counts) else float("nan")


def compare_top11(frame: pd.DataFrame, challenger: str, baseline: str) -> dict:
    """Paired, gameweek by gameweek: does `challenger` pick a better XI?"""
    a, b = _top11(frame, challenger), _top11(frame, baseline)
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
        "baseline_points_per_gw": float(b.mean()),
        "mean_gain_per_gw": float(diff.mean()),
        "median_gain_per_gw": float(np.median(diff)),
        "gameweeks_won": int((diff > 0).sum()),
        "gameweeks_lost": int((diff < 0).sum()),
        "season_gain_per_38_gw": float(diff.mean() * 38),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        # This project's rule: BOTH tests must pass before a claim is "proven".
        "significant": bool(diff.mean() > 0 and t_p < 0.05 and w_p < 0.05),
    }


def per_season(frame: pd.DataFrame, baseline: str) -> dict:
    """Four independent replications beat one pooled p-value."""
    return {
        season: compare_top11(frame[frame["season"] == season], "ours", baseline)
        for season in sorted(frame["season"].unique())
    }


def _verdict(vs_ppg: dict, seasons: dict) -> str:
    won = sum(1 for c in seasons.values() if c["mean_gain_per_gw"] > 0)
    n = len(seasons)
    sig = sum(1 for c in seasons.values() if c["significant"])
    gain, t_p, w_p = vs_ppg["mean_gain_per_gw"], vs_ppg["paired_t_p"], vs_ppg["wilcoxon_p"]

    replication = (
        f"The edge appears in **{won}/{n} seasons** independently "
        f"(significant on both tests in {sig}/{n})."
    )
    if gain > 0 and vs_ppg["significant"]:
        return (
            f"**The projection picks a better XI than the strongest naive baseline: "
            f"{gain:+.2f} points per gameweek ({vs_ppg['season_gain_per_38_gw']:+.0f} over a "
            f"38-gameweek season), winning {vs_ppg['gameweeks_won']}/{vs_ppg['gameweeks']} "
            f"gameweeks, significant on BOTH tests (t p={t_p:.4f}, Wilcoxon p={w_p:.4f}).** "
            f"{replication} This is a real, usable edge."
        )
    if gain > 0:
        return (
            f"**The projection picks a better XI ({gain:+.2f} points/GW, winning "
            f"{vs_ppg['gameweeks_won']}/{vs_ppg['gameweeks']} gameweeks) but the margin is NOT "
            f"significant on both tests (t p={t_p:.4f}, Wilcoxon p={w_p:.4f}). "
            f"Suggestive, not proven.** {replication}"
        )
    return (
        f"**The naive baseline picks a better XI ({gain:+.2f} points/GW).** "
        f"Points-per-gameweek silently encodes minutes, role and quality. "
        f"{replication} Recorded as a negative result."
    )


def build_report(frame, results, comparisons, seasons, run_id) -> str:
    ours, ppg = results["ours"], results["player_ppg"]
    vs_ppg = comparisons["player_ppg"]

    lines = [
        f"# FPL Projection Backtest (multi-season) - {run_id}",
        "",
        f"Walk-forward across **{frame['season'].nunique()} seasons**, "
        f"{frame.groupby(['season', 'gameweek']).ngroups} gameweeks, "
        f"{len(frame):,} player-gameweeks. To project gameweek k a model sees only "
        f"gameweeks 1..k-1 of that season and matches played before that gameweek's "
        f"first kickoff.",
        "",
        f"Seasons: {', '.join(sorted(frame['season'].unique()))}. Earlier seasons are "
        "excluded because FPL published no xG before 2022/23, and replaying them on "
        "realised goals would test a different model.",
        "",
        "**Handicap:** FPL publishes injury flags only for the current moment, not "
        "historically, so this backtest cannot use them. The live projection does. "
        "This therefore *understates* the real system.",
        "",
        "## Pooled results",
        "",
        pd.DataFrame(results).T.to_markdown(floatfmt=".4f"),
        "",
        "- `spearman_all` - rank correlation across **every** player, including the "
        "~700 fringe players who score 0-2. You never pick from that pool.",
        "- `spearman_pickable` - rank correlation restricted to players averaging "
        f"{PICKABLE_PPG}+ points so far (a prediction-time filter, no hindsight). "
        "This is the pool a manager actually chooses from.",
        "- `legal_xi_points_per_gw` - **the metric that decides.** Field the best XI "
        "this model could actually field (legal formation, max 3 per club) and sum "
        "what those players really scored.",
        "- `naive_top11_points_per_gw` / `naive_xi_goalkeepers` - the OLD, invalid "
        "metric and its tell. Ranking by projected points and taking the top 11 "
        "fields multiple goalkeepers; a manager fields one. Reported only to "
        "document why the earlier single-season result was wrong.",
        "",
        "## The decisive metric: who picks the better XI?",
        "",
        "| baseline | our XI | their XI | gain/GW | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for baseline, c in comparisons.items():
        lines.append(
            f"| {baseline} | {c['challenger_points_per_gw']:.2f} | "
            f"{c['baseline_points_per_gw']:.2f} | **{c['mean_gain_per_gw']:+.2f}** | "
            f"{c['gameweeks_won']}/{c['gameweeks']} | {c['paired_t_p']:.4f} | "
            f"{c['wilcoxon_p']:.4f} | {'yes' if c['significant'] else 'NO'} |"
        )

    lines += [
        "",
        "## Replication: ours vs `player_ppg`, season by season",
        "",
        "One pooled p-value can be driven by a single lucky season. These are "
        "independent replications.",
        "",
        "| season | our XI | ppg XI | gain/GW | GWs won | t p | Wilcoxon p | both? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for season, c in seasons.items():
        lines.append(
            f"| {season} | {c['challenger_points_per_gw']:.2f} | "
            f"{c['baseline_points_per_gw']:.2f} | **{c['mean_gain_per_gw']:+.2f}** | "
            f"{c['gameweeks_won']}/{c['gameweeks']} | {c['paired_t_p']:.4f} | "
            f"{c['wilcoxon_p']:.4f} | {'yes' if c['significant'] else 'NO'} |"
        )

    beats_rank_pool = ours["spearman_pickable"] > ppg["spearman_pickable"]
    lines += [
        "",
        "## Verdict",
        "",
        _verdict(vs_ppg, seasons),
        "",
        f"On error metrics the baselines win (MAE: ours {ours['mae']:.4f} vs "
        f"{ppg['mae']:.4f}; Spearman over all players: {ours['spearman_all']:.4f} vs "
        f"{ppg['spearman_all']:.4f}). That is expected and not a contradiction: those "
        f"metrics are dominated by hundreds of fringe players who reliably score ~0, "
        f"which a player's own average predicts almost perfectly. Restricted to the "
        f"**pickable pool**, the ranking gap "
        + (f"reverses in our favour (ours {ours['spearman_pickable']:.4f} vs "
           f"{ppg['spearman_pickable']:.4f})." if beats_rank_pool else
           f"does NOT reverse (ours {ours['spearman_pickable']:.4f} vs "
           f"{ppg['spearman_pickable']:.4f})."),
        "",
        "The engine's contribution is fixture information - clean-sheet probability "
        "and opponent strength - which changes nothing for a bench player who will "
        "score 0 either way, and matters most exactly at the top of the board where "
        "picks are made.",
        "",
        "## Caveats",
        "",
        "- The XI respects formation and the max-3-per-club rule but still ignores "
        "**budget**, and there is no captain (FPL doubles one player's score). All "
        "models face the same omission, so the comparison is fair.",
        "- Within a formation the XI is filled greedily by projected points subject "
        "to the club cap, not solved exactly. Applied identically to every model.",
        "- Injury flags unavailable historically; live they are used, which should "
        "favour our projection further.",
        "- `defensive_contribution` exists only in 2025/26; earlier seasons neither "
        "award nor project those points.",
        "- Double gameweeks are aggregated to one row per player-gameweek, so a "
        "player cannot be picked twice in one XI.",
        "- `price` is the price at that gameweek, not end-of-season.",
        "",
        "## Data source",
        "",
        "Per-gameweek history from the community FPL archive "
        "(https://github.com/vaastav/Fantasy-Premier-League, CC BY 4.0), which "
        "mirrors FPL's own `element-summary` endpoint. Team model trained on "
        "Understat match + xG data.",
    ]
    return "\n".join(lines)


def main():
    configure_logging(
        log_dir=Path(__file__).resolve().parents[2] / "logs",
        filename="benchmark_fpl_projections.log",
        level="INFO", max_bytes=5 * 1024 * 1024, backup_count=3,
    )

    frame = run_backtest()
    logger.info("scored %d player-gameweeks", len(frame))

    results = evaluate(frame)
    comparisons = {
        baseline: compare_top11(frame, "ours", baseline)
        for baseline in MODELS if baseline != "ours"
    }
    seasons = per_season(frame, "player_ppg")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = build_report(frame, results, comparisons, seasons, run_id)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"fpl_projection_backtest_multiseason_{run_id}.md").write_text(
        report, encoding="utf-8")
    (RESULTS_DIR / f"fpl_projection_backtest_multiseason_{run_id}.json").write_text(
        json.dumps({"results": results, "top11_comparisons": comparisons,
                    "per_season_vs_player_ppg": seasons}, indent=2), encoding="utf-8")
    print(report)
    return results


if __name__ == "__main__":
    main()
