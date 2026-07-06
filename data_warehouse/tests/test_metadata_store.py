import tempfile
import unittest
from pathlib import Path

from data_warehouse.ingest.metadata_store import (
    build_metadata,
    has_any_version,
    latest_version_metadata,
    metadata_path_for,
    new_version_id,
    read_latest_version,
    read_metadata,
    write_new_version,
)
from data_warehouse.utils.checksum import sha256_bytes


class TestMetadataStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.dataset_dir = Path(self._tmpdir.name) / "E0" / "2324"

    def test_no_version_exists_initially(self):
        self.assertFalse(has_any_version(self.dataset_dir))
        self.assertIsNone(read_latest_version(self.dataset_dir))
        self.assertIsNone(latest_version_metadata(self.dataset_dir, "2324.csv"))

    def test_write_new_version_creates_data_file_metadata_and_pointer(self):
        content = b"Div,Date\nE0,01/01/2024\n"
        metadata = build_metadata(
            source="football_data_co_uk",
            identifier="E0/2324",
            source_url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
            version=new_version_id(),
            local_path=self.dataset_dir / "2324.csv",
            content=content,
            checksum_sha256=sha256_bytes(content),
        )
        data_path = write_new_version(self.dataset_dir, "2324.csv", content, metadata)

        self.assertTrue(data_path.exists())
        self.assertEqual(data_path.read_bytes(), content)
        self.assertTrue(metadata_path_for(data_path).exists())

        self.assertTrue(has_any_version(self.dataset_dir))
        self.assertEqual(read_latest_version(self.dataset_dir), metadata.version)

        loaded = read_metadata(data_path)
        self.assertEqual(loaded.checksum_sha256, metadata.checksum_sha256)
        self.assertEqual(loaded.version, metadata.version)

    def test_second_version_advances_latest_pointer_and_keeps_first(self):
        content_v1 = b"Div,Date\nE0,01/01/2024\n"
        metadata_v1 = build_metadata(
            source="football_data_co_uk",
            identifier="E0/2324",
            source_url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
            version=new_version_id(),
            local_path=self.dataset_dir / "2324.csv",
            content=content_v1,
            checksum_sha256=sha256_bytes(content_v1),
        )
        write_new_version(self.dataset_dir, "2324.csv", content_v1, metadata_v1)

        content_v2 = b"Div,Date\nE0,01/01/2024\nE0,08/01/2024\n"
        metadata_v2 = build_metadata(
            source="football_data_co_uk",
            identifier="E0/2324",
            source_url="https://www.football-data.co.uk/mmz4281/2324/E0.csv",
            version=new_version_id(),
            local_path=self.dataset_dir / "2324.csv",
            content=content_v2,
            checksum_sha256=sha256_bytes(content_v2),
        )
        write_new_version(self.dataset_dir, "2324.csv", content_v2, metadata_v2)

        self.assertEqual(read_latest_version(self.dataset_dir), metadata_v2.version)
        self.assertNotEqual(metadata_v1.version, metadata_v2.version)

        # The first version's data file must still exist, untouched.
        first_version_path = self.dataset_dir / metadata_v1.version / "2324.csv"
        self.assertTrue(first_version_path.exists())
        self.assertEqual(first_version_path.read_bytes(), content_v1)

        latest = latest_version_metadata(self.dataset_dir, "2324.csv")
        self.assertEqual(latest.checksum_sha256, metadata_v2.checksum_sha256)

    def test_new_version_id_is_unique_across_rapid_calls(self):
        ids = {new_version_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


if __name__ == "__main__":
    unittest.main()
