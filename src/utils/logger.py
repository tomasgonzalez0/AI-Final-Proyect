"""Logging utilities for FoodVision AI."""

from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str = "foodvision", level: int = logging.INFO) -> logging.Logger:
    """
    Create or return a configured logger.

    Args:
        name (str): Logger name.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _has_stream_handler(logger):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _has_stream_handler(logger: logging.Logger) -> bool:
    """
    Check if the logger already has a stream handler.

    Args:
        logger (logging.Logger): Logger instance to inspect.

    Returns:
        bool: True when a stream handler is present.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            return True
    return False
