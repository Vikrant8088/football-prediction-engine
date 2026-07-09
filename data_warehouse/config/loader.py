"""Loads and validates data_warehouse configuration from YAML.

Configuration is parsed into typed objects rather than passed around as raw
dicts, so a missing or misspelled key fails fast at startup with a clear
error instead of surfacing as a confusing KeyError deep inside a source.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from data_warehouse.ingest.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class StorageConfig:
    raw_data_dir: Path


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path
    filename: str
    level: str
    max_bytes: int
    backup_count: int


@dataclass(frozen=True)
class HttpConfig:
    timeout_seconds: float
    max_retries: int
    backoff_factor: float
    request_delay_seconds: float
    user_agent: str


@dataclass(frozen=True)
class FootballDataCoUkConfig:
    base_url: str
    leagues: Dict[str, str]
    seasons: List[str]


@dataclass(frozen=True)
class UnderstatConfig:
    base_url: str
    request_headers: Dict[str, str]
    leagues: Dict[str, str]
    seasons: List[str]


@dataclass(frozen=True)
class ApiFootballConfig:
    base_url: str
    leagues: Dict[str, str]
    seasons: List[str]


@dataclass(frozen=True)
class AppConfig:
    storage: StorageConfig
    logging: LoggingConfig
    http: HttpConfig
    football_data_co_uk: FootballDataCoUkConfig
    understat: UnderstatConfig
    api_football: ApiFootballConfig
    repo_root: Path

    @property
    def raw_data_dir(self) -> Path:
        return self._resolve(self.storage.raw_data_dir)

    @property
    def log_dir(self) -> Path:
        return self._resolve(self.logging.log_dir)

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path


def _require(data: Dict[str, Any], key: str, context: str) -> Any:
    if key not in data:
        raise ConfigurationError(f"Missing required config key '{key}' in {context}")
    return data[key]


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH, repo_root: Path = None
) -> AppConfig:
    """Load, parse and validate the warehouse configuration file."""
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    storage_raw = _require(raw, "storage", "config.yaml")
    logging_raw = _require(raw, "logging", "config.yaml")
    http_raw = _require(raw, "http", "config.yaml")
    sources_raw = _require(raw, "sources", "config.yaml")
    fdc_raw = _require(sources_raw, "football_data_co_uk", "sources")
    understat_raw = _require(sources_raw, "understat", "sources")
    api_football_raw = _require(sources_raw, "api_football", "sources")

    storage = StorageConfig(
        raw_data_dir=Path(_require(storage_raw, "raw_data_dir", "storage")),
    )
    logging_config = LoggingConfig(
        log_dir=Path(_require(logging_raw, "log_dir", "logging")),
        filename=str(_require(logging_raw, "filename", "logging")),
        level=str(_require(logging_raw, "level", "logging")),
        max_bytes=int(_require(logging_raw, "max_bytes", "logging")),
        backup_count=int(_require(logging_raw, "backup_count", "logging")),
    )
    http = HttpConfig(
        timeout_seconds=float(_require(http_raw, "timeout_seconds", "http")),
        max_retries=int(_require(http_raw, "max_retries", "http")),
        backoff_factor=float(_require(http_raw, "backoff_factor", "http")),
        request_delay_seconds=float(
            _require(http_raw, "request_delay_seconds", "http")
        ),
        user_agent=str(_require(http_raw, "user_agent", "http")),
    )
    football_data_co_uk = FootballDataCoUkConfig(
        base_url=str(_require(fdc_raw, "base_url", "sources.football_data_co_uk")),
        leagues=dict(_require(fdc_raw, "leagues", "sources.football_data_co_uk")),
        seasons=list(_require(fdc_raw, "seasons", "sources.football_data_co_uk")),
    )
    understat = UnderstatConfig(
        base_url=str(_require(understat_raw, "base_url", "sources.understat")),
        request_headers=dict(
            _require(understat_raw, "request_headers", "sources.understat")
        ),
        leagues=dict(_require(understat_raw, "leagues", "sources.understat")),
        seasons=list(_require(understat_raw, "seasons", "sources.understat")),
    )
    api_football = ApiFootballConfig(
        base_url=str(_require(api_football_raw, "base_url", "sources.api_football")),
        leagues=dict(_require(api_football_raw, "leagues", "sources.api_football")),
        seasons=list(_require(api_football_raw, "seasons", "sources.api_football")),
    )

    resolved_root = repo_root if repo_root is not None else config_path.parents[2]

    return AppConfig(
        storage=storage,
        logging=logging_config,
        http=http,
        football_data_co_uk=football_data_co_uk,
        understat=understat,
        api_football=api_football,
        repo_root=resolved_root,
    )
