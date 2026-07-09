"""Abstract interface every raw data source must implement.

This is the sole extension point for adding new sources (StatsBomb,
ClubElo, API-Football, ...). The versioned raw-lake logic (skip / new
version / unchanged) lives here once, shared by every source; subclasses
only need to describe *what* is downloadable and *how* to reach it.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from data_warehouse.ingest.exceptions import DownloadError
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.metadata_store import (
    build_metadata,
    has_any_version,
    latest_version_metadata,
    new_version_id,
    write_new_version,
)
from data_warehouse.ingest.models import DatasetRef, DownloadOutcome
from data_warehouse.utils.checksum import sha256_bytes

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Base class for a single external raw data source."""

    name: str

    def __init__(self, raw_data_dir: Path, http_client: RateLimitedHttpClient):
        self._raw_data_dir = raw_data_dir
        self._http_client = http_client

    @property
    def request_headers(self) -> Dict[str, str]:
        """Extra HTTP headers this source requires on every request, merged
        over the shared client defaults. Empty for sources that need none;
        overridden by sources whose endpoint demands specific headers."""
        return {}

    @abstractmethod
    def list_available_datasets(self, **filters) -> List[DatasetRef]:
        """Return datasets this source is configured to know about.

        `filters` are source-specific keyword restrictions (e.g. this
        source's `leagues`/`seasons`); a source that doesn't recognize a
        given filter name should raise `ValueError`. Omitting all filters
        must return every configured dataset.
        """
        raise NotImplementedError

    def resolve_dataset_dir(self, ref: DatasetRef) -> Path:
        return self._raw_data_dir / self.name / ref.dataset_dir_relative

    def _fetch_content(self, ref: DatasetRef) -> bytes:
        """Fetch the raw bytes for one dataset. The default is a single GET;
        sources whose payload spans multiple requests (e.g. a paginated JSON
        API) override this to fetch and combine all pages into one payload."""
        return self._http_client.get_bytes(
            ref.source_url, extra_headers=self.request_headers
        )

    def download(self, ref: DatasetRef, force: bool = False) -> DownloadOutcome:
        """Fetch a dataset into the versioned raw lake.

        - If a version already exists and `force` is False: no network call
          is made; returns "skipped_exists".
        - If `force` is True (or no version exists yet): fetches the source,
          and either writes a new version (first time, or content changed)
          or returns "unchanged" if the content is byte-identical to the
          current latest version.
        """
        dataset_dir = self.resolve_dataset_dir(ref)

        if not force and has_any_version(dataset_dir):
            logger.info(
                "Skipping '%s': a version already exists at %s (pass force=True to check for updates)",
                ref.identifier,
                dataset_dir,
            )
            return DownloadOutcome(identifier=ref.identifier, status="skipped_exists")

        try:
            logger.info("Fetching '%s' from %s", ref.identifier, ref.source_url)
            content = self._fetch_content(ref)

            checksum = sha256_bytes(content)
            previous = latest_version_metadata(dataset_dir, ref.filename)
            if previous is not None and previous.checksum_sha256 == checksum:
                logger.info(
                    "'%s' unchanged since version '%s'; no new version created",
                    ref.identifier,
                    previous.version,
                )
                return DownloadOutcome(
                    identifier=ref.identifier, status="unchanged", metadata=previous
                )

            version = new_version_id()
            metadata = build_metadata(
                source=self.name,
                identifier=ref.identifier,
                source_url=ref.source_url,
                version=version,
                local_path=dataset_dir / version / ref.filename,
                content=content,
                checksum_sha256=checksum,
            )
            write_new_version(dataset_dir, ref.filename, content, metadata)
            return DownloadOutcome(
                identifier=ref.identifier, status="downloaded", metadata=metadata
            )
        except (DownloadError, OSError) as exc:
            logger.error("Failed to fetch '%s': %s", ref.identifier, exc)
            return DownloadOutcome(
                identifier=ref.identifier, status="failed", error=str(exc)
            )

    def download_all(self, force: bool = False) -> List[DownloadOutcome]:
        return [self.download(ref, force=force) for ref in self.list_available_datasets()]
