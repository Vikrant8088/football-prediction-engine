"""Registry mapping source names to their constructors.

Adding a new source (StatsBomb, ClubElo, API-Football) means writing a
`sources/<new_source>.py` module and adding one entry here - nothing else in
the CLI or ingest layer needs to change.
"""

from data_warehouse.config.loader import AppConfig
from data_warehouse.ingest.base_source import BaseDataSource
from data_warehouse.ingest.exceptions import UnknownSourceError
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.sources.api_football import ApiFootballInjuriesSource
from data_warehouse.sources.football_data_co_uk import FootballDataCoUkSource
from data_warehouse.sources.fpl import FplSource
from data_warehouse.sources.understat import UnderstatSource

SOURCE_NAMES = ("football_data_co_uk", "understat", "api_football", "fpl")


def build_source(
    name: str, config: AppConfig, http_client: RateLimitedHttpClient
) -> BaseDataSource:
    if name == "football_data_co_uk":
        return FootballDataCoUkSource(
            raw_data_dir=config.raw_data_dir,
            http_client=http_client,
            source_config=config.football_data_co_uk,
        )
    if name == "understat":
        return UnderstatSource(
            raw_data_dir=config.raw_data_dir,
            http_client=http_client,
            source_config=config.understat,
        )
    if name == "api_football":
        return ApiFootballInjuriesSource(
            raw_data_dir=config.raw_data_dir,
            http_client=http_client,
            source_config=config.api_football,
        )
    if name == "fpl":
        return FplSource(
            raw_data_dir=config.raw_data_dir,
            http_client=http_client,
            source_config=config.fpl,
        )
    raise UnknownSourceError(
        f"Unknown source '{name}'. Registered sources: {', '.join(SOURCE_NAMES)}"
    )
