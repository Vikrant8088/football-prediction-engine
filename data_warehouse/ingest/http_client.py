"""Thin, polite HTTP client wrapper used by all data sources.

Centralising this (rather than letting each source call `requests` directly)
means retry policy, timeouts, user-agent, and rate limiting are consistent
across every source and only need to be tuned in one place.
"""

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_warehouse.config.loader import HttpConfig
from data_warehouse.ingest.exceptions import DownloadError

logger = logging.getLogger(__name__)


class RateLimitedHttpClient:
    """A `requests.Session` wrapper with retries, timeout, and a minimum
    delay enforced between consecutive requests."""

    def __init__(self, config: HttpConfig):
        self._config = config
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.user_agent})

        retry = Retry(
            total=config.max_retries,
            backoff_factor=config.backoff_factor,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        self._last_request_monotonic: Optional[float] = None

    def _enforce_rate_limit(self) -> None:
        if self._last_request_monotonic is None:
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        remaining = self._config.request_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get_bytes(self, url: str) -> bytes:
        """Fetch a URL and return its raw response body as bytes.

        Raises `DownloadError` on any network failure or non-2xx response,
        after the underlying session has already retried transient failures.
        """
        self._enforce_rate_limit()
        try:
            response = self._session.get(url, timeout=self._config.timeout_seconds)
            self._last_request_monotonic = time.monotonic()
        except requests.RequestException as exc:
            self._last_request_monotonic = time.monotonic()
            raise DownloadError(f"Request failed for {url}: {exc}") from exc

        if not response.ok:
            raise DownloadError(
                f"Request to {url} returned HTTP {response.status_code}"
            )
        return response.content
