"""Data source for API-Football (api-sports.io direct endpoint) - injuries.

Fetches the per-fixture injury/suspension report ("who is missing for this
match") for a league-season from:

    {base_url}/injuries?league={id}&season={year}

Two things make this source different from the file-based ones, and both are
handled here rather than leaking into the shared base class:

1. Authentication by a SECRET key. The `x-apisports-key` header is required on
   every request; the key is read from the APIFOOTBALL_KEY environment variable
   at call time and is never written to config or committed to git.
2. Pagination. A season's injuries can span multiple pages, so `_fetch_content`
   walks every page and stores the combined list of injury records as one JSON
   payload in the versioned raw lake (the same immutable, checksummed,
   provenance-tracked storage every other source uses).

Free-plan note: API-Football restricts injury access to recent seasons (2022-24
at time of writing); an out-of-range season returns a plan-restriction error,
which is surfaced as a DownloadError rather than silently stored as empty.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from data_warehouse.config.loader import ApiFootballConfig
from data_warehouse.ingest.base_source import BaseDataSource
from data_warehouse.ingest.exceptions import DownloadError
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.models import DatasetRef

API_KEY_ENV = "APIFOOTBALL_KEY"
SUPPORTED_FILTERS = ("leagues", "seasons")


class ApiFootballInjuriesSource(BaseDataSource):
    name = "api_football"

    def __init__(
        self,
        raw_data_dir: Path,
        http_client: RateLimitedHttpClient,
        source_config: ApiFootballConfig,
    ):
        super().__init__(raw_data_dir=raw_data_dir, http_client=http_client)
        self._source_config = source_config

    @property
    def request_headers(self) -> Dict[str, str]:
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise DownloadError(
                f"Environment variable {API_KEY_ENV} is not set - it is required "
                f"to authenticate with API-Football (the key is a secret and is "
                f"never stored in config)."
            )
        return {"x-apisports-key": key}

    def list_available_datasets(self, **filters: Any) -> List[DatasetRef]:
        """One DatasetRef per (league, season): the season's full injury report."""
        unsupported = set(filters) - set(SUPPORTED_FILTERS)
        if unsupported:
            raise ValueError(
                f"Unsupported filters for {self.name}: {sorted(unsupported)}. "
                f"Supported: {SUPPORTED_FILTERS}"
            )

        selected_leagues = filters.get("leagues") or list(
            self._source_config.leagues.keys()
        )
        selected_seasons = filters.get("seasons") or list(self._source_config.seasons)

        refs = []
        for league in selected_leagues:
            for season in selected_seasons:
                url = f"{self._source_config.base_url}/injuries?league={league}&season={season}"
                refs.append(
                    DatasetRef(
                        identifier=f"{league}/{season}",
                        source_url=url,
                        dataset_dir_relative=f"injuries/{league}/{season}",
                        filename=f"injuries_{league}_{season}.json",
                    )
                )
        return refs

    def _fetch_content(self, ref: DatasetRef) -> bytes:
        """Return the season's injury records as one combined JSON document.

        The /injuries endpoint returns all records in a single response and
        rejects a `page` query param, so page 1 is fetched bare; a `page` param
        is only appended if the response itself declares more than one page
        (defensive - other API-Football endpoints do paginate this way)."""
        combined = []
        page = 1
        while True:
            url = ref.source_url if page == 1 else f"{ref.source_url}&page={page}"
            content = self._http_client.get_bytes(url, extra_headers=self.request_headers)
            payload = json.loads(content)

            errors = payload.get("errors")
            # API-Football returns errors as {} / [] when fine, or a populated
            # dict/list on failure (e.g. a free-plan season restriction).
            if errors:
                raise DownloadError(
                    f"API-Football returned errors for '{ref.identifier}': {errors}"
                )

            combined.extend(payload.get("response", []))
            paging = payload.get("paging") or {}
            total = int(paging.get("total", 1))
            if total <= 1 or int(paging.get("current", 1)) >= total:
                break
            page += 1

        return json.dumps(combined).encode("utf-8")
