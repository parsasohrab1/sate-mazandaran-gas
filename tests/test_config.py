"""Config loader tests — no heavy deps required."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from weld_defect.config import (
    TrainConfig, load_config, load_yaml, _resolve_env,
)


def test_resolve_env_uses_default(monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    assert _resolve_env("${NOPE_VAR,fallback}") == "fallback"


def test_resolve_env_uses_env(monkeypatch):
    monkeypatch.setenv("MY_VAR", "live")
    assert _resolve_env("${MY_VAR,x}") == "live"


def test_resolve_env_nested():
    out = _resolve_env({"a": ["${X,1}", "plain"], "b": {"c": "${Y,2}"}})
    assert out == {"a": ["1", "plain"], "b": {"c": "2"}}


def test_load_train_yaml_defaults():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "train.yaml", TrainConfig)
    assert cfg.data.num_classes == 6
    assert cfg.model.architecture == "ensemble"
    assert "resnet50" in cfg.model.members
    assert cfg.training.epochs >= 1
    assert cfg.metrics.primary == "f1_macro"
    assert len(cfg.data.class_names) == 6
    # class_weights aligns with num_classes
    assert len(cfg.data.class_weights) == cfg.data.num_classes


def test_load_yaml_missing_file():
    with pytest.raises(FileNotFoundError):
        load_yaml("/nonexistent/path.yaml")
