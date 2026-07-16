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


def score_squad(actuals: Dict[int, float], xi_ids: List[int], captain_id: int) -> float:
    """Actual FPL points of a starting XI, with the captain counted twice.

    A player absent from `actuals` scored nothing that gameweek (did not feature);
    v1 does not autosub him. `actuals` keys are FPL player ids.
    """
    total = sum(float(actuals.get(int(pid), 0.0)) for pid in xi_ids)
    total += float(actuals.get(int(captain_id), 0.0))     # captain: doubled
    return float(total)


def score_artifact(artifact: dict, actuals: Dict[int, float]) -> dict:
    """Score a locked gameweek artifact. Returns {gameweek, ours[, baseline, gain]}."""
    our_xi = [row["player_id"] for row in artifact["xi"]]
    ours = score_squad(actuals, our_xi, artifact["captain_id"])
    record = {"gameweek": int(artifact["gameweek"]), "ours": ours}
    baseline = artifact.get("baseline")
    if baseline is not None:
        record["baseline"] = score_squad(actuals, baseline["xi"], baseline["captain_id"])
        record["gain"] = ours - record["baseline"]
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
        paired = [r for r in self.records if "baseline" in r]
        base = {"season": self.season, "scored_gameweeks": len(self.records)}
        if not paired:
            base.update(paired_summary([], []))
            return base
        base.update(paired_summary([r["ours"] for r in paired],
                                   [r["baseline"] for r in paired]))
        return base

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


def gameweek_actuals(season: str, gameweek: int) -> Dict[int, float]:
    """Actual FPL points per player id for a completed gameweek, from the archive.

    Double gameweeks are summed, matching how the squad was scored. The community
    archive lags the live API by a few days; the live season's same-week scoring
    would read FPL's `event/{gw}/live` endpoint instead (a small follow-up), but the
    ledger machinery is identical either way.
    """
    from research.data.fpl_archive import load_gameweeks
    frame = load_gameweeks(season)
    week = frame[frame["gameweek"] == gameweek]
    return {int(pid): float(points)
            for pid, points in week.groupby("player_id")["total_points"].sum().items()}
