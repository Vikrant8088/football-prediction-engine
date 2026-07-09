"""Command-line entry point for the Football Data Warehouse.

Usage examples
--------------
Download everything configured for football-data.co.uk::

    python -m data_warehouse.cli download --source football_data_co_uk

Download only specific leagues/seasons::

    python -m data_warehouse.cli download --source football_data_co_uk \\
        --leagues E0 SP1 --seasons 2324 2223

Re-check datasets that already have a version on disk, creating a new
version only if the source content actually changed::

    python -m data_warehouse.cli download --source football_data_co_uk --force
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from data_warehouse.config.loader import load_config
from data_warehouse.ingest.http_client import RateLimitedHttpClient
from data_warehouse.ingest.models import DownloadOutcome
from data_warehouse.registry import SOURCE_NAMES, build_source
from data_warehouse.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_warehouse",
        description="Download historical football datasets into data/raw/.",
    )
    subparsers = parser.add_subparsers(dest="command")
    # Set as an attribute rather than via add_subparsers(required=True), which
    # is only accepted from Python 3.7 onward; this form works on 3.6 too.
    subparsers.required = True

    download_parser = subparsers.add_parser(
        "download", help="Download datasets from a configured source."
    )
    download_parser.add_argument(
        "--source",
        required=True,
        choices=SOURCE_NAMES,
        help="Which data source to download from.",
    )
    download_parser.add_argument(
        "--leagues",
        nargs="+",
        default=None,
        help="Restrict to specific league codes (source-dependent). "
        "Defaults to every league configured for the source.",
    )
    download_parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Restrict to specific season codes (source-dependent). "
        "Defaults to every season configured for the source.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch datasets that already have a version on disk, "
        "creating a new version only if the content has changed. "
        "Never mutates or deletes an existing version.",
    )
    return parser


def _summarize(outcomes: List[DownloadOutcome]) -> int:
    counts = {"downloaded": 0, "unchanged": 0, "skipped_exists": 0, "failed": 0}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1

    logger.info(
        "Done: %d downloaded (new version), %d unchanged, %d skipped (already existed), %d failed",
        counts["downloaded"],
        counts["unchanged"],
        counts["skipped_exists"],
        counts["failed"],
    )
    for outcome in outcomes:
        if outcome.status == "failed":
            logger.error("  FAILED: %s - %s", outcome.identifier, outcome.error)

    return 1 if counts["failed"] > 0 else 0


def run_download(args: argparse.Namespace) -> int:
    config = load_config()
    configure_logging(
        log_dir=config.log_dir,
        filename=config.logging.filename,
        level=config.logging.level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )

    http_client = RateLimitedHttpClient(config.http)
    source = build_source(args.source, config, http_client)

    filters = {}
    if args.leagues is not None:
        filters["leagues"] = args.leagues
    if args.seasons is not None:
        filters["seasons"] = args.seasons
    refs = source.list_available_datasets(**filters)

    outcomes = [source.download(ref, force=args.force) for ref in refs]
    return _summarize(outcomes)


def main(argv: List[str] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        return run_download(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
