"""Score a locked Bank-It squad against what really happened, and keep the ledger.

This is the second half of the forward-validation loop (docs/04_BANK_IT_PIPELINE.md
§4.6). `bank_it` locks a squad — ours and the `player_ppg` baseline — before the
deadline. After the gameweek resolves, this module reads each player's actual points
and scores both locked squads, captain doubled, exactly as FPL does. The per-gameweek
gain accumulates in a `SeasonLedger`, which reports the running paired test against
the pre-registered decision rule.

Scoring is a pure lookup because both squads were fully locked in the artifact: no
squad is ever reconstructed at scoring time, so hindsight cannot leak in.

Autosubs are not modelled (v1): a locked starter who plays 0 minutes scores 0 rather
than being replaced by a bench player. This is applied identically to both squads and
is stated in the design doc's caveats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

SEASON_GAMEWEEKS = 38

# The backtest scored from GW6 (`FIRST_SCORED_GAMEWEEK`): before that, no model has
# enough of the season to say anything, and the live engine's rates are cold-started
# from last season. The proven +3/GW therefore describes GW6-38, not GW1-38 — so the
# forward primary is scored over the same range. GW1-5 are still locked and scored, but
# reported separately as exploratory rather than folded into the pre-registered claim.
FIRST_VALIDATED_GAMEWEEK = 6


def _effective_captain(captain_id: int, vice_captain_id: Optional[int],
                       minutes: Optional[Dict[int, float]]) -> Optional[int]:
    """Whose score is doubled, applying FPL's vice-captain rule.

    If we know minutes and the captain played 0, the vice-captain is promoted; if
    the vice-captain also played 0, nobody is doubled. With no minutes information
    (`minutes` None) the captain is doubled unconditionally — the legacy behaviour,
    and what keeps this identical to the optimizer's `xi_actual_points`.
    """
    if minutes is None:
        return captain_id
    if float(minutes.get(int(captain_id), 0.0)) > 0:
        return captain_id
    if vice_captain_id is not None and float(minutes.get(int(vice_captain_id), 0.0)) > 0:
        return vice_captain_id
    return None                                    # captain and vice both blanked


def score_squad(points: Dict[int, float], xi_ids: List[int], captain_id: int,
                vice_captain_id: Optional[int] = None,
                minutes: Optional[Dict[int, float]] = None) -> float:
    """Actual FPL points of a starting XI, with the (effective) captain counted twice.

    A player absent from `points` scored nothing that gameweek (did not feature);
    v1 does not autosub him. `minutes`, when supplied, activates the vice-captain
    rule via `_effective_captain`. All keys are FPL player ids.
    """
    total = sum(float(points.get(int(pid), 0.0)) for pid in xi_ids)
    effective = _effective_captain(captain_id, vice_captain_id, minutes)
    if effective is not None:
        total += float(points.get(int(effective), 0.0))     # captaincy: doubled
    return float(total)


def _split_actuals(actuals: Dict[int, object]):
    """Accept either {id: points} or {id: {"points", "minutes"}}; return (points,
    minutes). minutes is None when no per-player minutes were supplied, which makes
    the scorer fall back to unconditional captain doubling (no vice-captain rule)."""
    points, minutes = {}, {}
    for pid, value in actuals.items():
        pid = int(pid)
        if isinstance(value, dict):
            points[pid] = float(value.get("points", 0.0))
            minutes[pid] = float(value.get("minutes", 0.0))
        else:
            points[pid] = float(value)
    return points, (minutes or None)


def score_artifact(artifact: dict, actuals: Dict[int, object]) -> dict:
    """Score a locked gameweek artifact. Returns {gameweek, ours[, baseline, gain]}.

    `actuals` may carry minutes (see `_split_actuals`); when it does, the vice-
    captain rule applies to both our squad and the baseline.
    """
    points, minutes = _split_actuals(actuals)
    our_xi = [row["player_id"] for row in artifact["xi"]]
    ours = score_squad(points, our_xi, artifact["captain_id"],
                       vice_captain_id=artifact.get("vice_captain_id"), minutes=minutes)
    record = {"gameweek": int(artifact["gameweek"]), "ours": ours}
    baseline = artifact.get("baseline")
    if baseline is not None:
        record["baseline"] = score_squad(
            points, baseline["xi"], baseline["captain_id"],
            vice_captain_id=baseline.get("vice_captain_id"), minutes=minutes)
        record["gain"] = ours - record["baseline"]

    # Declared variants (e.g. the Phase 6f predicted-lineup candidate). Each is scored
    # against the SAME actuals and reported both absolutely and as a gain over the
    # primary — the paired A/B that says whether the candidate earns its place.
    for name, block in (artifact.get("variants") or {}).items():
        if "xi" not in block:                  # e.g. a track that could not field an XI
            continue
        scored = score_squad(points, block["xi"], block["captain_id"],
                             vice_captain_id=block.get("vice_captain_id"),
                             minutes=minutes)
        # A carried-squad track can PAY to assemble its team (-4 per transfer beyond
        # the free ones). Scoring it gross would credit it with points it never had,
        # and would flatter exactly the policies that transfer most.
        hits = float(block.get("hits", 0.0))
        net = scored - hits
        record.setdefault("variants", {})[name] = {
            "points": net,
            "gross_points": scored,
            "hits": hits,
            "gain_vs_ours": net - ours,
        }
        if "baseline" in record:
            record["variants"][name]["gain_vs_baseline"] = net - record["baseline"]
    return record


def paired_summary(ours: List[float], baseline: List[float]) -> dict:
    """The pre-registered paired test: mean gain per gameweek, both p-values, and
    whether it clears the two-test rule. Shape matches `benchmark_fpl_optimizer`."""
    a, b = np.asarray(ours, float), np.asarray(baseline, float)
    diff = a - b
    n = len(diff)
    t_p = float(stats.ttest_rel(a, b)[1]) if n > 2 else float("nan")
    try:
        w_p = float(stats.wilcoxon(a, b)[1]) if n > 0 else float("nan")
    except ValueError:                       # all-zero differences etc.
        w_p = float("nan")
    mean_gain = float(diff.mean()) if n else float("nan")
    return {
        "gameweeks": int(n),
        "ours_per_gw": float(a.mean()) if n else float("nan"),
        "baseline_per_gw": float(b.mean()) if n else float("nan"),
        "mean_gain_per_gw": mean_gain,
        "median_gain_per_gw": float(np.median(diff)) if n else float("nan"),
        "gameweeks_won": int((diff > 0).sum()),
        "season_gain_per_38_gw": mean_gain * SEASON_GAMEWEEKS if n else float("nan"),
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        # This project's rule: BOTH tests must pass before a claim is "proven".
        "significant": bool(n > 0 and mean_gain > 0 and t_p < 0.05 and w_p < 0.05),
    }


class SeasonLedger:
    """The running record of the live season: one row per scored gameweek.

    Records are keyed by gameweek, so re-scoring a gameweek overwrites rather than
    duplicates. `summary()` runs the pre-registered paired test over every gameweek
    that has a baseline to compare against.
    """

    def __init__(self, season: str, records: Optional[List[dict]] = None):
        self.season = season
        self._records: Dict[int, dict] = {}
        for record in (records or []):
            self._records[int(record["gameweek"])] = record

    def add(self, record: dict) -> None:
        self._records[int(record["gameweek"])] = record

    @property
    def records(self) -> List[dict]:
        return [self._records[gw] for gw in sorted(self._records)]

    def summary(self) -> dict:
        """The PRE-REGISTERED primary: gameweeks from FIRST_VALIDATED_GAMEWEEK only.

        GW1-5 are deliberately excluded from the headline — the backtest never scored
        them, and the engine's rates there are cold-started from last season. They are
        still recorded and reported, via `exploratory_summary()`, but folding untested
        gameweeks into a claim the backtest never made would overstate it.
        """
        validated = [r for r in self.records
                     if int(r["gameweek"]) >= FIRST_VALIDATED_GAMEWEEK]
        paired = [r for r in validated if "baseline" in r]
        base = {
            "season": self.season,
            "scored_gameweeks": len(validated),
            "range": "GW%d-%d (the backtest's validated range)" % (
                FIRST_VALIDATED_GAMEWEEK, SEASON_GAMEWEEKS),
        }
        if not paired:
            base.update(paired_summary([], []))
        else:
            base.update(paired_summary([r["ours"] for r in paired],
                                       [r["baseline"] for r in paired]))
        base["variants"] = self.variant_summaries(records=validated)
        # The transfer A/B, reported at the top level because it is the open question
        # the live season exists to settle (see `carried`), not a footnote.
        base["carried_head_to_head"] = self.head_to_head(
            "carried_ours", "carried_ppg", records=validated)
        base["exploratory"] = self.exploratory_summary()
        return base

    def exploratory_summary(self) -> dict:
        """GW1-5, reported honestly and separately: outside the validated range, on
        cold-started rates. Evidence, never the headline."""
        early = [r for r in self.records
                 if int(r["gameweek"]) < FIRST_VALIDATED_GAMEWEEK and "baseline" in r]
        summary = {"range": "GW1-%d" % (FIRST_VALIDATED_GAMEWEEK - 1),
                   "status": "EXPLORATORY — outside the backtest's validated range; "
                             "rates cold-started from last season"}
        summary.update(paired_summary([r["ours"] for r in early],
                                      [r["baseline"] for r in early]))
        return summary

    def variant_summaries(self, records: Optional[List[dict]] = None) -> dict:
        """Each declared variant vs the PRIMARY, paired by gameweek — the forward A/B.

        Scored only on gameweeks where the variant was actually locked, so a feed that
        was missing for a week cannot silently borrow the primary's result.
        """
        records = self.records if records is None else records
        names = set()
        for record in records:
            names.update((record.get("variants") or {}).keys())

        summaries = {}
        for name in sorted(names):
            rows = [r for r in records if name in (r.get("variants") or {})]
            summaries[name] = paired_summary(
                [r["variants"][name]["points"] for r in rows],
                [r["ours"] for r in rows])
        return summaries

    def head_to_head(self, challenger: str, control: str,
                     records: Optional[List[dict]] = None) -> dict:
        """One variant against ANOTHER, paired by gameweek.

        `variant_summaries` compares every variant to the primary, which cannot answer
        the question the carried-squad tracks exist for: `carried_ours` versus
        `carried_ppg` — the same squad, the same policy, differing only in which
        projection chose the transfer. Scored only on gameweeks where BOTH were
        locked, so a week one track missed cannot be silently borrowed from the other.
        """
        records = self.records if records is None else records
        rows = [r for r in records
                if challenger in (r.get("variants") or {})
                and control in (r.get("variants") or {})]
        summary = {
            "challenger": challenger,
            "control": control,
            "paired_gameweeks": len(rows),
        }
        summary.update(paired_summary(
            [r["variants"][challenger]["points"] for r in rows],
            [r["variants"][control]["points"] for r in rows]))
        return summary

    def to_dict(self) -> dict:
        return {"season": self.season, "records": self.records, "summary": self.summary()}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("wrote ledger (%d gameweeks) to %s", len(self.records), path)

    @classmethod
    def load(cls, path) -> "SeasonLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["season"], payload.get("records"))


def gameweek_actuals(season: str, gameweek: int) -> Dict[int, dict]:
    """Actual points AND minutes per player id for a completed gameweek, from the
    archive: {player_id: {"points": p, "minutes": m}}.

    Minutes are needed for the vice-captain rule (a captain who played 0 is replaced
    by the vice-captain). Double gameweeks are summed, matching how the squad was
    scored. The community archive lags the live API by a few days; the live season's
    same-week scoring would read FPL's `event/{gw}/live` endpoint instead (a small
    follow-up), but the ledger machinery is identical either way.
    """
    from research.data.fpl_archive import load_gameweeks
    frame = load_gameweeks(season)
    week = frame[frame["gameweek"] == gameweek]
    grouped = week.groupby("player_id").agg({"total_points": "sum", "minutes": "sum"})
    return {int(pid): {"points": float(row["total_points"]), "minutes": float(row["minutes"])}
            for pid, row in grouped.iterrows()}
