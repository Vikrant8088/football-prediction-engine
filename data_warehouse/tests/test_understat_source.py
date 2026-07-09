"""Unit tests for the Understat source. The HTTP client is mocked, so these
never touch the network - they verify URL/layout construction, the required
XHR header, and the shared versioned-lake behavior inherited from
BaseDataSource."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from data_warehouse.config.loader import UnderstatConfig
from data_warehouse.sources.understat import UnderstatSource


class TestUnderstatSource(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.raw_data_dir = Path(self._tmp.name)
        self.http_client = MagicMock()
        self.config = UnderstatConfig(
            base_url="https://understat.com/getLeagueData",
            request_headers={"X-Requested-With": "XMLHttpRequest"},
            leagues={"EPL": "England - Premier League", "La_liga": "Spain - La Liga"},
            seasons=["2022", "2023"],
        )
        self.source = UnderstatSource(
            raw_data_dir=self.raw_data_dir,
            http_client=self.http_client,
            source_config=self.config,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_one_ref_per_league_season_pair(self):
        refs = self.source.list_available_datasets()
        self.assertEqual(len(refs), 4)  # 2 leagues x 2 seasons
        epl_2023 = next(r for r in refs if r.identifier == "EPL/2023")
        self.assertEqual(
            epl_2023.source_url, "https://understat.com/getLeagueData/EPL/2023"
        )
        self.assertEqual(epl_2023.dataset_dir_relative, "EPL/2023")
        self.assertEqual(epl_2023.filename, "EPL_2023.json")

    def test_filters_restrict_the_configured_set(self):
        refs = self.source.list_available_datasets(leagues=["EPL"], seasons=["2023"])
        self.assertEqual([r.identifier for r in refs], ["EPL/2023"])

    def test_unsupported_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            self.source.list_available_datasets(teams=["Arsenal"])

    def test_download_sends_required_xhr_header(self):
        self.http_client.get_bytes.return_value = b'{"dates": []}'
        ref = self.source.list_available_datasets(leagues=["EPL"], seasons=["2023"])[0]

        outcome = self.source.download(ref)

        self.assertEqual(outcome.status, "downloaded")
        self.http_client.get_bytes.assert_called_once_with(
            ref.source_url, extra_headers={"X-Requested-With": "XMLHttpRequest"}
        )
        version_data_path = (
            self.source.resolve_dataset_dir(ref)
            / outcome.metadata.version
            / "EPL_2023.json"
        )
        self.assertTrue(version_data_path.exists())


if __name__ == "__main__":
    unittest.main()
