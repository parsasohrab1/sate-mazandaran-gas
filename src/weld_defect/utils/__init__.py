"""Structured logging and reproducibility helpers."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def get_logger(name: str = "weld_defect") -> logging.Logger:
    """Return a configured logger. Idempotent across imports."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
