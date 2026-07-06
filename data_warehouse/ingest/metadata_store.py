"""Manages the versioned raw data lake layout.

Layout, per dataset::

    <raw_root>/<source>/<dataset_dir_relative>/
        latest.json                 # {"latest_version": "<version>"}
        <version>/
            <filename>
            <filename>.meta.json

Each version directory is written once and never mutated - a "new" fetch
either creates a new version (if content changed, or none exists yet) or is a
no-op (if content is byte-identical to the latest version). This gives every
downstream consumer (normalize/validate/canonical stages, or a human
debugging a backtest) a reproducible, point-in-time snapshot to pin against,
rather than a single mutable file that silently changes under them.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data_warehouse.ingest.models import DatasetMetadata

logger = logging.getLogger(__name__)

METADATA_SUFFIX = ".meta.json"
LATEST_POINTER_FILENAME = "latest.json"


def new_version_id() -> str:
    """A sortable, filesystem-safe version identifier based on UTC time plus
    a short random suffix, so two processes writing at the same microsecond
    (e.g. concurrent CLI invocations) still get distinct version ids."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _atomic_write_text(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: write to a temp file in the same
    directory, then rename over the destination. A process killed mid-write
    leaves the temp file, never a truncated `path`, so readers of `path`
    never observe partial content."""
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _latest_pointer_path(dataset_dir: Path) -> Path:
    return dataset_dir / LATEST_POINTER_FILENAME


def read_latest_version(dataset_dir: Path) -> Optional[str]:
    pointer_path = _latest_pointer_path(dataset_dir)
    if not pointer_path.exists():
        return None
    return json.loads(pointer_path.read_text(encoding="utf-8"))["latest_version"]


def _write_latest_pointer(dataset_dir: Path, version: str) -> None:
    pointer_path = _latest_pointer_path(dataset_dir)
    _atomic_write_text(
        pointer_path, json.dumps({"latest_version": version}, indent=2)
    )


def metadata_path_for(data_file_path: Path) -> Path:
    return data_file_path.with_name(data_file_path.name + METADATA_SUFFIX)


def latest_version_metadata(
    dataset_dir: Path, filename: str
) -> Optional[DatasetMetadata]:
    """Return the metadata of the current latest version, if any exists."""
    version = read_latest_version(dataset_dir)
    if version is None:
        return None
    data_file_path = dataset_dir / version / filename
    return read_metadata(data_file_path)


def has_any_version(dataset_dir: Path) -> bool:
    return read_latest_version(dataset_dir) is not None


def build_metadata(
    source: str,
    identifier: str,
    source_url: str,
    version: str,
    local_path: Path,
    content: bytes,
    checksum_sha256: str,
) -> DatasetMetadata:
    """Build metadata for a version about to be written.

    `checksum_sha256` is taken as a parameter (rather than computed here)
    because the caller typically already needed it to decide whether the
    content changed since the last version - recomputing it a second time
    over the same bytes would be wasted work.
    """
    return DatasetMetadata(
        source=source,
        identifier=identifier,
        source_url=source_url,
        version=version,
        local_path=str(local_path),
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        file_size_bytes=len(content),
        checksum_sha256=checksum_sha256,
    )


def write_new_version(
    dataset_dir: Path,
    filename: str,
    content: bytes,
    metadata: DatasetMetadata,
) -> Path:
    """Write a brand-new, immutable version directory and advance the
    latest-version pointer. Returns the path to the written data file."""
    version_dir = dataset_dir / metadata.version
    version_dir.mkdir(parents=True, exist_ok=False)

    data_file_path = version_dir / filename
    data_file_path.write_bytes(content)

    meta_path = metadata_path_for(data_file_path)
    _atomic_write_text(
        meta_path, json.dumps(metadata.to_dict(), indent=2, sort_keys=True)
    )

    _write_latest_pointer(dataset_dir, metadata.version)
    logger.info(
        "Wrote new version '%s' for dataset '%s' (%d bytes)",
        metadata.version,
        metadata.identifier,
        len(content),
    )
    return data_file_path


def read_metadata(data_file_path: Path) -> Optional[DatasetMetadata]:
    meta_path = metadata_path_for(data_file_path)
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return DatasetMetadata.from_dict(data)
