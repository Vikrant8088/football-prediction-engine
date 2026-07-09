"""End-to-end integration test for the Phase 1 pipeline.

Unlike the unit tests (which isolate one component), this drives the whole
chain the way `benchmark_xg.main` does, minus the network:

    ingest storage layer  ->  xg_loader  ->  models (incl. PoissonXG)
      ->  run_walk_forward  ->  summarize  ->  pairwise_significance

A small synthetic Understat league is written into a temp raw lake using the
REAL metadata_store writer (so the storage/loader contract is exercised, not
mocked), then read back and run through the actual benchmark harness. Any
break at a seam between these subsystems fails here even when every unit test
passes.

Dixon-Coles is excluded only for speed; the goals-vs-xG contrast (poisson vs
poisson_xg) plus the baseline floor are what this integration needs to prove
flow correctly through the harness.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import research.data.xg_loader as xg_loader
from data_warehouse.ingest.metadata_store import (
    build_metadata,
    new_version_id,
    write_new_version,
)
from data_warehouse.utils.checksum import sha256_bytes
from research.data.xg_loader import load_understat_matches
from research.evaluation.benchmark import run_walk_forward, summarize
from research.evaluation.significance import pairwise_significance
from research.experiments.baseline import BaselineFrequencyModel
from research.experiments.poisson import PoissonModel
from research.experiments.poisson_xg import PoissonXGModel

# A clean, well-separated synthetic league: strength drives both xG and goals,
# so a fitted model should comfortably beat the frequency baseline.
TEAMS = ["Alpha", "Bravo", "Charlie", "Delta"]
STRENGTH = {"Alpha": 2.2, "Bravo": 1.6, "Charlie": 1.1, "Delta": 0.6}
MODELS = {
    "baseline": BaselineFrequencyModel,
    "poisson": PoissonModel,
    "poisson_xg": PoissonXGModel,
}


def _season_payload(start_year: str):
    """A full double round-robin (each ordered pair once) for one season,
    plus one unplayed fixture to confirm the loader skips it."""
    base = datetime(int(start_year), 8, 1, 15, 0, 0)
    dates = []
    n = 0
    for home in TEAMS:
        for away in TEAMS:
            if home == away:
                continue
            home_xg = round(STRENGTH[home] * 1.15, 3)
            away_xg = round(STRENGTH[away] * 0.9, 3)
            dt = base + timedelta(days=3 * n)
            dates.append({
                "id": f"{start_year}-{n}",
                "isResult": True,
                "h": {"title": home},
                "a": {"title": away},
                "goals": {"h": str(int(round(home_xg))), "a": str(int(round(away_xg)))},
                "xG": {"h": str(home_xg), "a": str(away_xg)},
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            })
            n += 1
    dates.append({  # unplayed fixture, must be ignored by the loader
        "id": f"{start_year}-future",
        "isResult": False,
        "h": {"title": "Alpha"}, "a": {"title": "Bravo"},
        "datetime": (base + timedelta(days=400)).strftime("%Y-%m-%d %H:%M:%S"),
    })
    return {"teams": {}, "players": [], "dates": dates}


class TestBenchmarkXgIntegration(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw = Path(self._tmp.name)
        self.seasons = ["2010", "2011", "2012"]
        for year in self.seasons:
            self._ingest_via_storage_layer("EPL", year, _season_payload(year))

    def tearDown(self):
        self._tmp.cleanup()

    def _ingest_via_storage_layer(self, league, year, payload):
        """Write a season through the real versioned-lake writer, exactly as
        the ingest download path does."""
        content = json.dumps(payload).encode("utf-8")
        filename = f"{league}_{year}.json"
        dataset_dir = self.raw / "understat" / league / year
        version = new_version_id()
        metadata = build_metadata(
            source="understat",
            identifier=f"{league}/{year}",
            source_url="https://test.local",
            version=version,
            local_path=dataset_dir / version / filename,
            content=content,
            checksum_sha256=sha256_bytes(content),
        )
        write_new_version(dataset_dir, filename, content, metadata)

    def _load(self):
        fake_cfg = SimpleNamespace(
            raw_data_dir=self.raw,
            understat=SimpleNamespace(seasons=self.seasons),
        )
        with patch.object(xg_loader, "load_config", return_value=fake_cfg):
            return load_understat_matches("EPL")

    def test_loader_reads_back_what_storage_wrote(self):
        matches = self._load()
        # 12 played matches per season x 3 seasons; unplayed fixtures dropped.
        self.assertEqual(len(matches), 36)
        self.assertEqual(matches["season"].nunique(), 3)
        self.assertTrue(matches["date"].is_monotonic_increasing)
        for col in ("home_xg", "away_xg", "home_goals", "away_goals", "result"):
            self.assertIn(col, matches.columns)

    def test_full_walk_forward_pipeline(self):
        matches = self._load()
        predictions, runtimes = run_walk_forward(
            matches, model_builders=MODELS, min_training_seasons=2
        )

        # One eval season (2012-13), 12 matches, for each of 3 models.
        self.assertEqual(set(predictions["model"]), set(MODELS))
        self.assertEqual(len(predictions), 12 * len(MODELS))

        probs = predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())

        # Every model recorded measured runtime for the single fold.
        for name in MODELS:
            self.assertEqual(runtimes[name]["n_folds"], 1)

    def test_summary_and_significance_are_well_formed(self):
        matches = self._load()
        predictions, runtimes = run_walk_forward(
            matches, model_builders=MODELS, min_training_seasons=2
        )
        summary = summarize(predictions, runtimes)

        for col in ("log_loss", "rps", "brier_score", "n_predictions"):
            self.assertIn(col, summary.columns)
        self.assertTrue(np.isfinite(summary["log_loss"]).all())

        # On clean, strength-driven data, a fitted model must beat the
        # frequency baseline - a real end-to-end signal that models learned.
        self.assertLess(
            summary.loc["poisson_xg", "log_loss"],
            summary.loc["baseline", "log_loss"],
        )

        significance = pairwise_significance(predictions, list(MODELS))
        self.assertEqual(len(significance), 3)  # 3 choose 2 pairs
        for col in ("paired_t_pvalue", "wilcoxon_pvalue"):
            vals = significance[col].to_numpy()
            self.assertTrue(np.all((vals >= 0) & (vals <= 1) | np.isnan(vals)))


if __name__ == "__main__":
    unittest.main()
