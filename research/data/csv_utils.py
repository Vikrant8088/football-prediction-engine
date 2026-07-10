"""Resilient CSV reading for the raw lake.

football-data.co.uk does not publish its CSVs in a single consistent encoding.
Most are UTF-8 (some with a BOM), but a few - notably the Scottish and Italian
second-tier files - contain cp1252 bytes such as 0xA0 (non-breaking space) that
are invalid UTF-8. Hard-coding `utf-8-sig` makes those seasons unreadable, which
silently drops whole leagues from an analysis.

We try encodings in order of fidelity: real UTF-8 first (so genuinely UTF-8
files are never mangled), then cp1252, then latin-1 as the last resort - latin-1
maps every byte to a character, so it always succeeds and guarantees we never
lose a file to an encoding error. The team names we depend on are ASCII, so a
late-fallback decode cannot corrupt a join key.
"""

import io
from pathlib import Path
from typing import Union

import pandas as pd

ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def read_csv_resilient(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV, trying progressively more permissive encodings.

    The bytes are read once and decoded in memory rather than re-opening the
    file per encoding: a failed `pd.read_csv` can leave the file handle open,
    which on Windows blocks the file from being deleted afterwards.
    """
    data = Path(path).read_bytes()

    last_error = None
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        return pd.read_csv(io.StringIO(text), **kwargs)

    raise ValueError(
        f"Could not decode {path} with any of {ENCODINGS}"
    ) from last_error
