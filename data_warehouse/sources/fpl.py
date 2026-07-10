"""Data source for the official Fantasy Premier League API.

Free, public, and unauthenticated:

    {base_url}/bootstrap-static/   players, teams, positions, gameweeks
    {base_url}/fixtures/           every fixture with its gameweek

`bootstrap-static` is unusually rich: alongside FPL's own scoring stats it
carries Opta-derived per-player `expected_goals`, `expected_assists` and
`expected_goals_conceded`, plus availability (`status`,
`chance_of_playing_next_round`) and price. That is the entire input side of a
fantasy projection, for nothing.

The endpoint rejects the warehouse's default bot user-agent, so a browser one is
supplied from config (never hardcoded here). Payloads are stored verbatim in the
versioned raw lake; parsing happens later, research-side.
"""

from pathlib import Path
from typing import Any, Dict, List

from data_warehouse.config.loader import FplConfig
from data_warehouse.ingest.base_source import BaseDataSource
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.models import DatasetRef

SUPPORTED_FILTERS = ("datasets",)


class FplSource(BaseDataSource):
    name = "fpl"

    def __init__(
        self,
        raw_data_dir: Path,
        http_client: RateLimitedHttpClient,
        source_config: FplConfig,
    ):
        super().__init__(raw_data_dir=raw_data_dir, http_client=http_client)
        self._source_config = source_config

    @property
    def request_headers(self) -> Dict[str, str]:
        return dict(self._source_config.request_headers)

    def list_available_datasets(self, **filters: Any) -> List[DatasetRef]:
        unsupported = set(filters) - set(SUPPORTED_FILTERS)
        if unsupported:
            raise ValueError(
                f"Unsupported filters for {self.name}: {sorted(unsupported)}. "
                f"Supported: {SUPPORTED_FILTERS}"
            )

        selected = filters.get("datasets") or list(self._source_config.datasets.keys())

        refs = []
        for dataset in selected:
            refs.append(
                DatasetRef(
                    identifier=dataset,
                    source_url=f"{self._source_config.base_url}/{dataset}/",
                    dataset_dir_relative=dataset,
                    filename=f"{dataset}.json",
                )
            )
        return refs
