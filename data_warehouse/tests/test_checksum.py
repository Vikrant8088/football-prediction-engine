import hashlib
import unittest

from data_warehouse.utils.checksum import sha256_bytes


class TestChecksum(unittest.TestCase):
    def test_matches_hashlib_reference(self):
        data = b"date,home,away\n2024-01-01,Arsenal,Chelsea\n"
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(sha256_bytes(data), expected)

    def test_different_content_gives_different_checksum(self):
        self.assertNotEqual(sha256_bytes(b"a"), sha256_bytes(b"b"))


if __name__ == "__main__":
    unittest.main()
