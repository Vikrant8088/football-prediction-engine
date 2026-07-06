"""Data source implementation for football-data.co.uk.

football-data.co.uk publishes one CSV per league per season at:

    {base_url}/{season}/{league_code}.csv

e.g. https://www.football-data.co.uk/mmz4281/2324/E0.csv for the 2023-24
English Premier League. Both the season codes and league codes are
configuration, not code (see config/config.yaml) - this module only knows
the URL pattern and directory layout.
"""

from pathlib import Path
from typing import Any, List

from data_warehouse.config.loader import FootballDataCoUkConfig
from data_warehouse.ingest.base_source import BaseDataSource
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.models import DatasetRef

SUPPORTED_FILTERS = ("leagues", "seasons")


class FootballDataCoUkSource(BaseDataSource):
    name = "football_data_co_uk"

    def __init__(
        self,
        raw_data_dir: Path,
        http_client: RateLimitedHttpClient,
        source_config: FootballDataCoUkConfig,
    ):
        super().__init__(raw_data_dir=raw_data_dir, http_client=http_client)
        self._source_config = source_config

    def list_available_datasets(self, **filters: Any) -> List[DatasetRef]:
        """Return one DatasetRef per (league, season) pair.

        Accepts optional `leagues=[...]` / `seasons=[...]` keyword filters to
        restrict the configured full set (e.g. for a CLI invocation
        targeting specific leagues only). Any other filter name is rejected.
        """
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
        for league_code in selected_leagues:
            for season in selected_seasons:
                url = f"{self._source_config.base_url}/{season}/{league_code}.csv"
                refs.append(
                    DatasetRef(
                        identifier=f"{league_code}/{season}",
                        source_url=url,
                        dataset_dir_relative=f"{league_code}/{season}",
                        filename=f"{season}.csv",
                    )
                )
        return refs
