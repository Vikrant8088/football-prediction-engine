"""Close the forward-validation loop: score a locked gameweek and update the ledger.

`bank_it` locks a squad before the deadline; `scorer` knows how to score a locked
artifact and keep a `SeasonLedger`; `fpl_live` fetches the actuals the day the
gameweek finishes. Nothing yet joins them into the once-a-week action a live season
needs, which left the loop half-built: we could lock squads but not actually run the
running comparison they exist to feed. This is that missing half.

Once per gameweek, after it is `data_checked`:

    load the locked artifact  ->  score it against FPL live actuals  ->  add to the
    ledger  ->  save  ->  report the running pre-registered test

Two integrity properties, both inherited rather than re-implemented:

  - the squad is NEVER reconstructed here; only the *locked* artifact is scored, so
    hindsight cannot leak in (this is `scorer`'s guarantee);
  - the actuals are refused unless FPL has confirmed the gameweek's bonus, so a
    provisional score cannot be written into the ledger as if it were final (this is
    `fpl_live`'s guarantee).

The result is that running this is safe to automate: it either scores a genuinely
final gameweek or declines, and it can be re-run without corrupting the ledger
because records are keyed by gameweek and overwrite rather than duplicate.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

from prediction_engine.fpl.scorer import SeasonLedger, score_artifact

logger = logging.getLogger(__name__)

LIVE_RESULTS_DIR = Path(__file__).resolve().parents[2] / "research" / "results" / "live"


def artifact_path(season: str, gameweek: int, base: Path = None) -> Path:
    base = base or LIVE_RESULTS_DIR
    return base / str(season) / ("GW%02d.json" % int(gameweek))


def ledger_path(season: str, base: Path = None) -> Path:
    base = base or LIVE_RESULTS_DIR
    return base / str(season) / "ledger.json"


def load_artifact(season: str, gameweek: int, base: Path = None) -> dict:
    path = artifact_path(season, gameweek, base)
    if not path.exists():
        raise FileNotFoundError(
            "no locked artifact at %s — was the squad banked before the deadline?"
            % path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger(season: str, base: Path = None) -> SeasonLedger:
    path = ledger_path(season, base)
    if path.exists():
        return SeasonLedger.load(path)
    return SeasonLedger(season)


def score_gameweek(season: str, gameweek: int,
                   actuals: Optional[Dict[int, dict]] = None,
                   require_final: bool = True, base: Path = None,
                   persist: bool = True) -> dict:
    """Score one locked gameweek into the season ledger.

    `actuals` is normally fetched from FPL live; it can be injected (a completed
    season, a test) to run without the network. Returns {record, summary} where
    `record` is this gameweek's line and `summary` is the running pre-registered test.
    """
    artifact = load_artifact(season, gameweek, base)
    if actuals is None:
        from research.data.fpl_live import live_actuals
        actuals = live_actuals(gameweek, require_final=require_final)

    record = score_artifact(artifact, actuals)
    ledger = load_ledger(season, base)
    ledger.add(record)
    if persist:
        ledger.save(ledger_path(season, base))
        logger.info("scored GW%d: ours=%.1f%s", gameweek, record["ours"],
                    (" baseline=%.1f gain=%+.1f" % (record["baseline"], record["gain"]))
                    if "baseline" in record else "")
    return {"record": record, "summary": ledger.summary()}


def _format_summary(result: dict) -> str:
    record, summary = result["record"], result["summary"]
    lines = ["Scored GW%d." % record["gameweek"]]
    if "baseline" in record:
        lines.append("  ours %.1f vs baseline %.1f  (gain %+.1f this week)"
                     % (record["ours"], record["baseline"], record["gain"]))
    for name, block in (record.get("variants") or {}).items():
        extra = (" [%d hits]" % block["hits"]) if block.get("hits") else ""
        lines.append("  variant %-14s %.1f%s" % (name, block["points"], extra))

    lines.append("")
    lines.append("Running primary (%s): %d scored gameweeks, gain %+.2f/GW, "
                 "t p=%.4f, W p=%.4f -> %s"
                 % (summary.get("range", "?"), summary.get("scored_gameweeks", 0),
                    summary.get("mean_gain_per_gw", float("nan")),
                    summary.get("paired_t_p", float("nan")),
                    summary.get("wilcoxon_p", float("nan")),
                    "SIGNIFICANT" if summary.get("significant") else "not yet significant"))

    h2h = summary.get("carried_head_to_head")
    if h2h and h2h.get("paired_gameweeks"):
        lines.append("Transfer A/B (carried_ours vs carried_ppg): %+.2f/GW over %d GW, "
                     "t p=%.4f" % (h2h["mean_gain_per_gw"], h2h["paired_gameweeks"],
                                   h2h.get("paired_t_p", float("nan"))))
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prediction_engine.fpl.score_live",
        description="Score a locked gameweek against FPL live actuals and update the "
                    "season ledger.")
    parser.add_argument("--season", required=True, help="e.g. 2026-27")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--provisional", action="store_true",
                        help="score even if FPL has not confirmed the gameweek's bonus "
                             "(NEVER for the real ledger; the numbers may still change)")
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    try:
        result = score_gameweek(args.season, args.gameweek,
                                require_final=not args.provisional)
    except Exception as exc:                       # a clear message beats a traceback
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
