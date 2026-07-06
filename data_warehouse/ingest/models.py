"""Data models shared across all data sources.

These types form the contract between `BaseDataSource` implementations and the
rest of the warehouse (versioned raw storage, CLI, logging). A new source
must be able to express its work purely in terms of these models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DatasetRef:
    """A reference to one downloadable dataset, before it has been fetched.

    `identifier` must be stable and unique within a source (e.g. "E0/2324").
    `dataset_dir_relative` is the directory (relative to the source's raw
    root) under which every historical version of this dataset is stored;
    `filename` is the name given to the payload inside each version
    directory (see `core/metadata_store.py` for the versioned layout).
    """

    identifier: str
    source_url: str
    dataset_dir_relative: str
    filename: str


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata recorded alongside every raw snapshot version."""

    source: str
    identifier: str
    source_url: str
    version: str
    local_path: str
    downloaded_at_utc: str
    file_size_bytes: int
    checksum_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "source_url": self.source_url,
            "version": self.version,
            "local_path": self.local_path,
            "downloaded_at_utc": self.downloaded_at_utc,
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        return cls(
            source=data["source"],
            identifier=data["identifier"],
            source_url=data["source_url"],
            version=data["version"],
            local_path=data["local_path"],
            downloaded_at_utc=data["downloaded_at_utc"],
            file_size_bytes=data["file_size_bytes"],
            checksum_sha256=data["checksum_sha256"],
        )


@dataclass(frozen=True)
class DownloadOutcome:
    """Result of attempting to fetch a single dataset into the raw lake.

    - "downloaded": a new version was fetched and written (first time, or
      content changed since the last version).
    - "unchanged": a re-fetch was requested (force=True) but the content was
      byte-identical to the latest existing version, so no new version was
      created.
    - "skipped_exists": a version already exists and force=False, so no
      network request was made at all.
    - "failed": the fetch attempt errored.
    """

    identifier: str
    status: str  # "downloaded" | "unchanged" | "skipped_exists" | "failed"
    metadata: Optional[DatasetMetadata] = None
    error: Optional[str] = None
