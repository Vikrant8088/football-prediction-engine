import unittest

from data_warehouse.config.loader import load_config


class TestConfigLoader(unittest.TestCase):
    def test_loads_default_config(self):
        config = load_config()

        self.assertEqual(config.storage.raw_data_dir.as_posix(), "data/raw")

        self.assertEqual(config.logging.filename, "data_warehouse.log")
        self.assertEqual(config.logging.level, "INFO")
        self.assertGreater(config.logging.max_bytes, 0)
        self.assertGreater(config.logging.backup_count, 0)

        self.assertGreater(config.http.timeout_seconds, 0)
        self.assertGreater(config.http.max_retries, 0)

        self.assertIn("E0", config.football_data_co_uk.leagues)
        self.assertIn("2324", config.football_data_co_uk.seasons)
        self.assertTrue(config.football_data_co_uk.base_url.startswith("https://"))

    def test_raw_data_dir_resolves_relative_to_repo_root(self):
        config = load_config()
        self.assertTrue(config.raw_data_dir.is_absolute())
        self.assertEqual(config.raw_data_dir, config.repo_root / "data" / "raw")

    def test_log_dir_resolves_relative_to_repo_root(self):
        config = load_config()
        self.assertTrue(config.log_dir.is_absolute())
        self.assertEqual(config.log_dir, config.repo_root / "logs")


if __name__ == "__main__":
    unittest.main()
