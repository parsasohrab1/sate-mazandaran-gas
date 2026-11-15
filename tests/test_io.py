"""Tests for utils.io."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from weld_defect.utils.io import ensure_dir, load_json, load_yaml, save_json, save_yaml


def test_ensure_dir_creates(tmp_path):
    """ensure_dir creates nested directories."""
    p = ensure_dir(tmp_path / "a" / "b" / "c")
    assert p.exists()
    assert p.is_dir()


def test_ensure_dir_idempotent(tmp_path):
    """ensure_dir on existing dir is a no-op."""
    p = ensure_dir(tmp_path)
    p2 = ensure_dir(tmp_path)
    assert p == p2


def test_save_and_load_json(tmp_path):
    """save_json + load_json roundtrip."""
    data = {"a": 1, "b": [1, 2, 3], "c": {"d": "hello"}}
    p = save_json(data, tmp_path / "data.json")
    assert p.exists()
    loaded = load_json(p)
    assert loaded == data


def test_save_json_creates_parent(tmp_path):
    """save_json creates parent directories."""
    save_json({"x": 1}, tmp_path / "nested" / "dir" / "f.json")
    assert (tmp_path / "nested" / "dir" / "f.json").exists()


def test_save_and_load_yaml(tmp_path):
    """save_yaml + load_yaml roundtrip."""
    data = {"name": "test", "values": [1, 2, 3], "nested": {"k": "v"}}
    p = save_yaml(data, tmp_path / "data.yaml")
    assert p.exists()
    loaded = load_yaml(p)
    assert loaded == data


def test_load_json_missing_raises(tmp_path):
    """load_json raises for missing file."""
    with pytest.raises(FileNotFoundError):
        load_json(tmp_path / "nonexistent.json")


def test_load_yaml_missing_raises(tmp_path):
    """load_yaml raises for missing file."""
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yaml")
