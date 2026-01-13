from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str = __name__, level: int = logging.INFO, stream: Optional[object] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream or sys.stdout)
        formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
