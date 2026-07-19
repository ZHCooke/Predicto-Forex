"""Single place to configure logging. Call `setup_logging()` from entrypoints."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config import LOG_DIR

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, logfile: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(LOG_DIR) / logfile, encoding="utf-8"))

    logging.basicConfig(
        level=level, format=_FORMAT, handlers=handlers, datefmt="%Y-%m-%d %H:%M:%S"
    )
