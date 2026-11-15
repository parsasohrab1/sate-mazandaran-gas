"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from weld_defect.config import (
    TrainConfig,
    load_config,
    load_eval_config,
    load_yaml,
)


def test_load_train_config():
    """load_config returns a TrainConfig with expected fields."""
    cfg = load_config("configs/train.yaml")
    assert isinstance(cfg, TrainConfig)
    assert cfg.experiment.name == "weld-defect-ensemble-v1"
    assert cfg.experiment.seed == 42
    assert cfg.data.num_classes == 6
    assert len(cfg.data.class_names) == 6
    assert "no_defect" in cfg.data.class_names
    assert "crack" in cfg.data.class_names
    assert cfg.data.class_weights == [1.0, 1.4, 1.6, 1.5, 1.2, 1.3]
    assert cfg.model.architecture == "ensemble"
    assert cfg.training.epochs == 40
    assert cfg.training.batch_size == 32
    assert cfg.loss.name == "focal_loss"
    assert cfg.loss.gamma == 2.0
    assert cfg.metrics.primary == "f1_macro"


def test_load_eval_config():
    """load_eval_config returns an EvalConfig with acceptance gates."""
    cfg = load_eval_config("configs/eval.yaml")
    assert cfg.model.architecture == "ensemble"
    assert cfg.model.num_classes == 6
    assert cfg.acceptance.min_f1_macro == 0.85
    assert cfg.acceptance.min_recall_crack == 0.95
    assert cfg.acceptance.min_precision_no_defect == 0.90
    assert cfg.acceptance.max_inference_ms == 120
    assert cfg.acceptance.require_expert_review is True
    assert cfg.eval.calibration == "temperature_scaling"


def test_env_var_substitution(monkeypatch):
    """${VAR,default} placeholders are substituted from env vars."""
    monkeypatch.setenv("TEST_MLFLOW_URI", "file:/tmp/test_mlruns")
    yaml_text = "experiment:\n  mlflow_tracking_uri: ${TEST_MLFLOW_URI,file:./default}\n  seed: 1\n"
    p = Path("configs/_test_env.yaml")
    p.write_text(yaml_text, encoding="utf-8")
    try:
        raw = load_yaml(p)
        assert raw["experiment"]["mlflow_tracking_uri"] == "file:/tmp/test_mlruns"
    finally:
        p.unlink()


def test_env_var_default():
    """${VAR,default} falls back to default when env var is not set."""
    import os
    os.environ.pop("NONEXISTENT_VAR_XYZ", None)
    yaml_text = "x: ${NONEXISTENT_VAR_XYZ,fallback_value}\n"
    p = Path("configs/_test_default.yaml")
    p.write_text(yaml_text, encoding="utf-8")
    try:
        raw = load_yaml(p)
        assert raw["x"] == "fallback_value"
    finally:
        p.unlink()


def test_missing_config_raises():
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent.yaml")
