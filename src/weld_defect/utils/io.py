"""Safe I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as JSON to ``path`` (pretty-printed)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    return p


def load_json(path: str | Path) -> Any:
    """Load JSON from ``path``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as YAML to ``path``."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)
    return p


def load_yaml(path: str | Path) -> Any:
    """Load YAML from ``path``."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
