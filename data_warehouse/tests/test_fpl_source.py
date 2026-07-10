"""Unit tests for the FPL source. HTTP is mocked - no network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from data_warehouse.config.loader import FplConfig
from data_warehouse.sources.fpl import FplSource

BROWSER_UA = "Mozilla/5.0 (test)"


class TestFplSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.http_client = MagicMock()
        self.config = FplConfig(
            base_url="https://fantasy.premierleague.com/api",
            request_headers={"User-Agent": BROWSER_UA},
            datasets={"bootstrap-static": "players", "fixtures": "fixtures"},
        )
        self.source = FplSource(
            raw_data_dir=Path(self._tmp.name),
            http_client=self.http_client,
            source_config=self.config,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_one_ref_per_dataset(self):
        refs = self.source.list_available_datasets()
        self.assertEqual({r.identifier for r in refs}, {"bootstrap-static", "fixtures"})
        boot = next(r for r in refs if r.identifier == "bootstrap-static")
        self.assertEqual(
            boot.source_url, "https://fantasy.premierleague.com/api/bootstrap-static/"
        )
        self.assertEqual(boot.filename, "bootstrap-static.json")

    def test_filter_restricts_datasets(self):
        refs = self.source.list_available_datasets(datasets=["fixtures"])
        self.assertEqual([r.identifier for r in refs], ["fixtures"])

    def test_unsupported_filter_rejected(self):
        with self.assertRaises(ValueError):
            self.source.list_available_datasets(seasons=["2024"])

    def test_download_sends_browser_user_agent(self):
        # The FPL endpoint rejects the default bot user-agent.
        self.http_client.get_bytes.return_value = json.dumps({"elements": []}).encode()
        ref = self.source.list_available_datasets(datasets=["fixtures"])[0]

        outcome = self.source.download(ref)

        self.assertEqual(outcome.status, "downloaded")
        self.http_client.get_bytes.assert_called_once_with(
            ref.source_url, extra_headers={"User-Agent": BROWSER_UA}
        )


if __name__ == "__main__":
    unittest.main()
