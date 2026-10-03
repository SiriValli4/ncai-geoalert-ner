"""Logging setup. Call `configure_logging()` once at startup; everywhere else
use `logging.getLogger(__name__)` -- never `print()`.
"""

from __future__ import annotations

import logging.config


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                },
            },
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default"},
            },
            "root": {"handlers": ["console"], "level": level.upper()},
            "loggers": {
                # uvicorn installs its own handlers; let them propagate to root instead.
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
            },
        }
    )
