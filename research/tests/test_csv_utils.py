"""Tests for resilient CSV reading.

Regression guard: football-data.co.uk's Scottish and Serie B files contain
cp1252 bytes (e.g. 0xA0) that are invalid UTF-8. Hard-coding utf-8 silently
dropped those whole leagues from an analysis.
"""

import tempfile
import unittest
from pathlib import Path

from research.data.csv_utils import read_csv_resilient


class TestReadCsvResilient(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_reads_plain_utf8(self):
        path = self._write("a.csv", b"HomeTeam,FTHG\nArsenal,2\n")
        df = read_csv_resilient(path)
        self.assertEqual(df.iloc[0]["HomeTeam"], "Arsenal")

    def test_strips_utf8_bom(self):
        path = self._write("b.csv", "﻿HomeTeam,FTHG\nArsenal,2\n".encode("utf-8"))
        df = read_csv_resilient(path)
        self.assertIn("HomeTeam", df.columns)  # BOM not glued onto the first header

    def test_reads_cp1252_bytes_that_are_invalid_utf8(self):
        # 0xA0 is a cp1252 non-breaking space and an invalid UTF-8 start byte.
        path = self._write("c.csv", b"HomeTeam,FTHG\nCelt\xa0ic,2\n")
        df = read_csv_resilient(path)  # must not raise
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["FTHG"], 2)

    def test_utf8_is_preferred_over_fallbacks(self):
        # A genuine UTF-8 accented name must decode as UTF-8, not be mangled by
        # a premature latin-1 fallback.
        path = self._write("d.csv", "HomeTeam,FTHG\nAtlético,2\n".encode("utf-8"))
        df = read_csv_resilient(path)
        self.assertEqual(df.iloc[0]["HomeTeam"], "Atlético")


if __name__ == "__main__":
    unittest.main()
