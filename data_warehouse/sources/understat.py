"""Data source implementation for Understat (understat.com).

Understat publishes per-league, per-season match data - crucially including
Expected Goals (xG) for both teams - as JSON from an XHR endpoint:

    {base_url}/{league}/{start_year}

e.g. https://understat.com/getLeagueData/EPL/2023 for the 2023/24 English
Premier League. The response is a JSON object with `dates` (one entry per
match, carrying goals, xG and kickoff time), `teams`, and `players`. The
raw JSON payload is stored verbatim in the versioned lake; extracting the
match-level xG fields is a later, research-side transform (see
research/data/xg_loader.py), keeping ingestion faithful to the source.

Both the league identifiers (Understat's own "EPL"/"La_liga"/... codes) and
the season identifiers (the calendar year a season started) are configuration,
not code - this module only knows the URL pattern and directory layout.
"""

from pathlib import Path
from typing import Any, Dict, List

from data_warehouse.config.loader import UnderstatConfig
from data_warehouse.ingest.base_source import BaseDataSource
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.models import DatasetRef

SUPPORTED_FILTERS = ("leagues", "seasons")


class UnderstatSource(BaseDataSource):
    name = "understat"

    def __init__(
        self,
        raw_data_dir: Path,
        http_client: RateLimitedHttpClient,
        source_config: UnderstatConfig,
    ):
        super().__init__(raw_data_dir=raw_data_dir, http_client=http_client)
        self._source_config = source_config

    @property
    def request_headers(self) -> Dict[str, str]:
        # The getLeagueData endpoint returns 404 unless this header is sent;
        # the exact header set lives in config, not hardcoded here.
        return dict(self._source_config.request_headers)

    def list_available_datasets(self, **filters: Any) -> List[DatasetRef]:
        """Return one DatasetRef per (league, season) pair.

        Accepts optional `leagues=[...]` / `seasons=[...]` keyword filters to
        restrict the configured full set. Any other filter name is rejected.
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
        for league in selected_leagues:
            for season in selected_seasons:
                url = f"{self._source_config.base_url}/{league}/{season}"
                refs.append(
                    DatasetRef(
                        identifier=f"{league}/{season}",
                        source_url=url,
                        dataset_dir_relative=f"{league}/{season}",
                        filename=f"{league}_{season}.json",
                    )
                )
        return refs
