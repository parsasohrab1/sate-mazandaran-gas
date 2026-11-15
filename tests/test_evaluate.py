"""Tests for evaluate.py and utils.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from weld_defect.utils.metrics import check_acceptance, compute_metrics, save_metrics_report


def test_compute_metrics_perfect():
    """Perfect predictions give accuracy=1."""
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([0, 1, 2, 3, 4, 5])
    m = compute_metrics(y_true, y_pred, class_names=["a", "b", "c", "d", "e", "f"])
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_compute_metrics_with_errors():
    """Metrics reflect prediction errors."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(5 / 6, abs=0.01)
    assert "confusion_matrix" in m
    assert "per_class" in m
    assert len(m["per_class"]) == 3


def test_compute_metrics_with_probs():
    """roc_auc_ovr is computed when y_prob is provided."""
    # Need enough samples per class for roc_auc to work.
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2])
    # Soft probabilities with 3 classes.
    rng = np.random.RandomState(0)
    y_prob = rng.dirichlet([1, 1, 1], size=12)
    m = compute_metrics(y_true, y_pred, y_prob=y_prob, class_names=["a", "b", "c"])
    # roc_auc_ovr may be None if sklearn can't compute it; just check the key exists.
    assert "roc_auc_ovr" in m


def test_check_acceptance_all_pass():
    """All gates pass when metrics exceed thresholds."""
    metrics = {
        "f1_macro": 0.90,
        "per_class": [
            {"class_name": "no_defect", "precision": 0.95, "recall": 0.90},
            {"class_name": "porosity", "precision": 0.88, "recall": 0.85},
            {"class_name": "crack", "precision": 0.92, "recall": 0.97},
            {"class_name": "lack_of_fusion", "precision": 0.85, "recall": 0.83},
            {"class_name": "undercut", "precision": 0.84, "recall": 0.82},
            {"class_name": "slag_inclusion", "precision": 0.86, "recall": 0.84},
        ],
        "inference_ms_avg": 50.0,
    }
    gates = {
        "min_f1_macro": 0.85,
        "min_recall_crack": 0.95,
        "min_precision_no_defect": 0.90,
        "max_inference_ms": 120,
        "require_expert_review": True,
    }
    result = check_acceptance(metrics, gates)
    assert result["pass"] is True
    assert all(c["pass"] for c in result["checks"].values())


def test_check_acceptance_f1_fails():
    """F1 below threshold fails."""
    metrics = {
        "f1_macro": 0.70,
        "per_class": [
            {"class_name": "no_defect", "precision": 0.95, "recall": 0.90},
            {"class_name": "porosity", "precision": 0.88, "recall": 0.85},
            {"class_name": "crack", "precision": 0.92, "recall": 0.97},
            {"class_name": "lack_of_fusion", "precision": 0.85, "recall": 0.83},
            {"class_name": "undercut", "precision": 0.84, "recall": 0.82},
            {"class_name": "slag_inclusion", "precision": 0.86, "recall": 0.84},
        ],
        "inference_ms_avg": 50.0,
    }
    gates = {
        "min_f1_macro": 0.85,
        "min_recall_crack": 0.95,
        "min_precision_no_defect": 0.90,
        "max_inference_ms": 120,
        "require_expert_review": True,
    }
    result = check_acceptance(metrics, gates)
    assert result["pass"] is False
    assert result["checks"]["f1_macro"]["pass"] is False


def test_check_acceptance_crack_recall_fails():
    """Low crack recall fails the safety-critical gate."""
    metrics = {
        "f1_macro": 0.90,
        "per_class": [
            {"class_name": "no_defect", "precision": 0.95, "recall": 0.90},
            {"class_name": "porosity", "precision": 0.88, "recall": 0.85},
            {"class_name": "crack", "precision": 0.92, "recall": 0.80},  # too low
            {"class_name": "lack_of_fusion", "precision": 0.85, "recall": 0.83},
            {"class_name": "undercut", "precision": 0.84, "recall": 0.82},
            {"class_name": "slag_inclusion", "precision": 0.86, "recall": 0.84},
        ],
        "inference_ms_avg": 50.0,
    }
    gates = {
        "min_f1_macro": 0.85,
        "min_recall_crack": 0.95,
        "min_precision_no_defect": 0.90,
        "max_inference_ms": 120,
        "require_expert_review": True,
    }
    result = check_acceptance(metrics, gates)
    assert result["pass"] is False
    assert result["checks"]["recall_crack"]["pass"] is False


def test_save_metrics_report(tmp_path):
    """save_metrics_report writes the expected files."""
    metrics = {
        "accuracy": 0.9,
        "f1_macro": 0.85,
        "per_class": [
            {"class_name": "a", "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10},
        ],
        "confusion_matrix": [[8, 2], [1, 9]],
    }
    save_metrics_report(metrics, tmp_path / "report")
    assert (tmp_path / "report" / "metrics.json").exists()
    assert (tmp_path / "report" / "per_class_report.txt").exists()


@pytest.mark.slow
def test_evaluate_smoke(tmp_path):
    """evaluate() runs end-to-end on tiny synthetic data with a trained checkpoint."""
    from weld_defect.data.synthetic import generate_synthetic_dataset
    from weld_defect.evaluate import evaluate
    from weld_defect.models.factory import build_model, save_model

    syn = generate_synthetic_dataset(n_samples=24, out_dir=tmp_path / "syn", seed=42, size=32)

    # Save a tiny model.
    model = build_model("custom_cnn", num_classes=6, pretrained=False)
    ckpt = tmp_path / "model.pt"
    save_model(model, ckpt)

    config_text = f"""
model:
  checkpoint: {ckpt}
  architecture: custom_cnn
  input_size: 32
  num_classes: 6
data:
  test_split: {syn}/test.csv
  image_dir: {syn}/images
  batch_size: 4
  num_workers: 0
  class_names: [no_defect, porosity, crack, lack_of_fusion, undercut, slag_inclusion]
eval:
  thresholds:
    no_defect: 0.5
    porosity: 0.45
    crack: 0.55
    lack_of_fusion: 0.5
    undercut: 0.5
    slag_inclusion: 0.48
  top_k: 1
  calibration: none
metrics:
  primary: f1_macro
  also: [accuracy]
  confusion_matrix: true
  per_class_report: true
  roc_curves: false
acceptance:
  min_f1_macro: 0.0
  min_recall_crack: 0.0
  min_precision_no_defect: 0.0
  max_inference_ms: 1000
  require_expert_review: true
artifacts:
  report_dir: {tmp_path}/metrics
  mlflow_run: link
  save_misclassified: true
  error_analysis_json: {tmp_path}/error_analysis.json
"""
    cfg_path = tmp_path / "eval.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    result = evaluate(str(cfg_path))
    assert "metrics" in result
    assert "acceptance" in result
    assert (tmp_path / "metrics" / "metrics.json").exists()
