"""Unit tests for the API-Football injuries source. The HTTP client is mocked,
so these never hit the network or need a real key - they verify URL/layout
construction, the secret-key header (from the env var), and multi-page
combination."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from data_warehouse.config.loader import ApiFootballConfig
from data_warehouse.ingest.exceptions import DownloadError
from data_warehouse.sources.api_football import API_KEY_ENV, ApiFootballInjuriesSource


def _page(records, current, total):
    return json.dumps(
        {"errors": [], "paging": {"current": current, "total": total}, "response": records}
    ).encode("utf-8")


class TestApiFootballInjuriesSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.http_client = MagicMock()
        self.config = ApiFootballConfig(
            base_url="https://v3.football.api-sports.io",
            leagues={"39": "England - Premier League"},
            seasons=["2022", "2023"],
        )
        self.source = ApiFootballInjuriesSource(
            raw_data_dir=Path(self._tmp.name),
            http_client=self.http_client,
            source_config=self.config,
        )
        self._prev_key = os.environ.get(API_KEY_ENV)
        os.environ[API_KEY_ENV] = "test-secret-key"

    def tearDown(self):
        self._tmp.cleanup()
        if self._prev_key is None:
            os.environ.pop(API_KEY_ENV, None)
        else:
            os.environ[API_KEY_ENV] = self._prev_key

    def test_lists_one_ref_per_league_season(self):
        refs = self.source.list_available_datasets()
        self.assertEqual(len(refs), 2)
        r = next(x for x in refs if x.identifier == "39/2022")
        self.assertEqual(
            r.source_url, "https://v3.football.api-sports.io/injuries?league=39&season=2022"
        )
        self.assertEqual(r.dataset_dir_relative, "injuries/39/2022")
        self.assertEqual(r.filename, "injuries_39_2022.json")

    def test_missing_key_raises(self):
        os.environ.pop(API_KEY_ENV, None)
        ref = self.source.list_available_datasets(seasons=["2022"])[0]
        outcome = self.source.download(ref)
        self.assertEqual(outcome.status, "failed")
        self.assertIn(API_KEY_ENV, outcome.error)

    def test_download_combines_pages_and_sends_key(self):
        # Two pages of injury records get combined into one stored payload.
        self.http_client.get_bytes.side_effect = [
            _page([{"player": {"id": 1}}, {"player": {"id": 2}}], current=1, total=2),
            _page([{"player": {"id": 3}}], current=2, total=2),
        ]
        ref = self.source.list_available_datasets(leagues=["39"], seasons=["2022"])[0]

        outcome = self.source.download(ref)

        self.assertEqual(outcome.status, "downloaded")
        # Two pages requested, both carrying the secret key header.
        self.assertEqual(self.http_client.get_bytes.call_count, 2)
        for call in self.http_client.get_bytes.call_args_list:
            # call[1] is the kwargs dict (call.kwargs only exists on Python 3.8+).
            self.assertEqual(
                call[1]["extra_headers"], {"x-apisports-key": "test-secret-key"}
            )
        stored = json.loads(
            (
                self.source.resolve_dataset_dir(ref)
                / outcome.metadata.version
                / "injuries_39_2022.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual([r["player"]["id"] for r in stored], [1, 2, 3])

    def test_api_error_becomes_failed_outcome(self):
        self.http_client.get_bytes.return_value = json.dumps(
            {"errors": {"plan": "Free plans do not have access to this season"}, "response": []}
        ).encode("utf-8")
        ref = self.source.list_available_datasets(leagues=["39"], seasons=["2022"])[0]

        outcome = self.source.download(ref)
        self.assertEqual(outcome.status, "failed")
        self.assertIn("plan", outcome.error)


if __name__ == "__main__":
    unittest.main()
