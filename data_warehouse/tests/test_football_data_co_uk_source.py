import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from data_warehouse.config.loader import FootballDataCoUkConfig
from data_warehouse.sources.football_data_co_uk import FootballDataCoUkSource


class TestFootballDataCoUkSource(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.raw_data_dir = Path(self._tmpdir.name)

        self.source_config = FootballDataCoUkConfig(
            base_url="https://www.football-data.co.uk/mmz4281",
            leagues={"E0": "Premier League", "SP1": "La Liga"},
            seasons=["2324", "2223"],
        )
        self.http_client = MagicMock()
        self.source = FootballDataCoUkSource(
            raw_data_dir=self.raw_data_dir,
            http_client=self.http_client,
            source_config=self.source_config,
        )

    def test_list_available_datasets_builds_expected_urls_and_paths(self):
        refs = self.source.list_available_datasets()
        self.assertEqual(len(refs), 4)  # 2 leagues x 2 seasons

        by_identifier = {ref.identifier: ref for ref in refs}
        ref = by_identifier["E0/2324"]
        self.assertEqual(
            ref.source_url, "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
        )
        self.assertEqual(ref.dataset_dir_relative, "E0/2324")
        self.assertEqual(ref.filename, "2324.csv")

    def test_list_available_datasets_can_be_restricted(self):
        refs = self.source.list_available_datasets(leagues=["E0"], seasons=["2324"])
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].identifier, "E0/2324")

    def test_list_available_datasets_rejects_unsupported_filters(self):
        with self.assertRaises(ValueError):
            self.source.list_available_datasets(teams=["Arsenal"])

    def test_download_writes_a_new_version_and_advances_latest_pointer(self):
        self.http_client.get_bytes.return_value = b"Div,Date\nE0,01/01/2024\n"
        ref = self.source.list_available_datasets(leagues=["E0"], seasons=["2324"])[0]

        outcome = self.source.download(ref)

        self.assertEqual(outcome.status, "downloaded")
        dataset_dir = self.source.resolve_dataset_dir(ref)
        version_data_path = dataset_dir / outcome.metadata.version / "2324.csv"
        self.assertTrue(version_data_path.exists())
        self.assertEqual(version_data_path.read_bytes(), b"Div,Date\nE0,01/01/2024\n")
        self.http_client.get_bytes.assert_called_once_with(ref.source_url)

        # The recorded local_path must point at the actual version directory,
        # not the dataset directory (which never itself holds the data file).
        self.assertEqual(Path(outcome.metadata.local_path), version_data_path)

    def test_download_skips_existing_dataset_without_calling_http_client(self):
        self.http_client.get_bytes.return_value = b"first content"
        ref = self.source.list_available_datasets(leagues=["E0"], seasons=["2324"])[0]

        first = self.source.download(ref)
        self.assertEqual(first.status, "downloaded")

        second = self.source.download(ref)
        self.assertEqual(second.status, "skipped_exists")
        self.http_client.get_bytes.assert_called_once()  # not called again

    def test_download_force_with_identical_content_reports_unchanged(self):
        ref = self.source.list_available_datasets(leagues=["E0"], seasons=["2324"])[0]

        self.http_client.get_bytes.return_value = b"same content"
        first = self.source.download(ref)
        first_version = first.metadata.version

        second = self.source.download(ref, force=True)

        self.assertEqual(second.status, "unchanged")
        self.assertEqual(self.http_client.get_bytes.call_count, 2)
        dataset_dir = self.source.resolve_dataset_dir(ref)
        # No new version directory should have been created.
        version_dirs = [p.name for p in dataset_dir.iterdir() if p.is_dir()]
        self.assertEqual(version_dirs, [first_version])

    def test_download_force_with_changed_content_creates_new_version(self):
        ref = self.source.list_available_datasets(leagues=["E0"], seasons=["2324"])[0]

        self.http_client.get_bytes.return_value = b"first content"
        first = self.source.download(ref)

        self.http_client.get_bytes.return_value = b"updated content"
        second = self.source.download(ref, force=True)

        self.assertEqual(second.status, "downloaded")
        self.assertNotEqual(second.metadata.version, first.metadata.version)

        dataset_dir = self.source.resolve_dataset_dir(ref)
        version_dirs = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
        self.assertEqual(version_dirs, {first.metadata.version, second.metadata.version})

        first_data = dataset_dir / first.metadata.version / "2324.csv"
        second_data = dataset_dir / second.metadata.version / "2324.csv"
        self.assertEqual(first_data.read_bytes(), b"first content")
        self.assertEqual(second_data.read_bytes(), b"updated content")


if __name__ == "__main__":
    unittest.main()
