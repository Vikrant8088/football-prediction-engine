"""Central logging configuration for the Football Data Warehouse.

Logging is configured once, at process start (CLI entry point or test setup),
and every module simply calls `logging.getLogger(__name__)`. This keeps log
formatting and destinations in one place rather than scattered per-module.
"""

import logging
import logging.handlers
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(
    log_dir: Path,
    filename: str,
    level: str,
    max_bytes: int,
    backup_count: int,
) -> None:
    """Configure the root logger with console + rotating file handlers.

    Safe to call multiple times (e.g. across tests) - existing handlers on
    the root logger are cleared first to avoid duplicate log lines.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.getLevelName(level))
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
