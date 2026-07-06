"""Checksum utilities.

football-data.co.uk does not publish checksums for its files, so any checksum
recorded by this warehouse is one we computed ourselves at download time. It
should be treated as a fingerprint for detecting future drift or corruption,
not as third-party-verified integrity data.
"""

import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()
