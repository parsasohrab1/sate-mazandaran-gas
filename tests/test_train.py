"""Tests for train.py."""

from __future__ import annotations

import pytest
import torch

from weld_defect.train import FocalLoss


def test_focal_loss_shape():
    """FocalLoss returns a scalar for batched input."""
    loss_fn = FocalLoss(gamma=2.0, alpha=[1.0, 1.4, 1.6, 1.5, 1.2, 1.3])
    logits = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0


def test_focal_loss_reduces():
    """FocalLoss with reduction='sum' returns a sum."""
    loss_fn = FocalLoss(gamma=2.0, reduction="sum")
    logits = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_focal_loss_no_alpha():
    """FocalLoss without alpha works."""
    loss_fn = FocalLoss(gamma=2.0, alpha=None)
    logits = torch.randn(2, 6)
    targets = torch.tensor([0, 1])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_focal_loss_zero_for_perfect_pred():
    """FocalLoss approaches 0 for confident correct predictions."""
    loss_fn = FocalLoss(gamma=0.0)  # gamma=0 → standard CE
    logits = torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0, -100.0]])
    targets = torch.tensor([0])
    loss = loss_fn(logits, targets)
    assert loss.item() < 0.01


@pytest.mark.slow
def test_train_smoke(tmp_path):
    """train() runs end-to-end on tiny synthetic data."""
    from weld_defect.data.synthetic import generate_synthetic_dataset
    from weld_defect.train import train

    # Generate tiny dataset.
    syn = generate_synthetic_dataset(n_samples=24, out_dir=tmp_path / "syn", seed=42, size=32)

    # Write a minimal config.
    config_text = f"""
experiment:
  name: smoke-test
  seed: 42
  mlflow_tracking_uri: file:{tmp_path}/mlruns
  mlflow_experiment: smoke-test
  mlflow_model_name: smoke-test
data:
  root: {tmp_path}
  train_split: {syn}/train.csv
  val_split: {syn}/val.csv
  test_split: {syn}/test.csv
  image_dir: {syn}/images
  input_size: 32
  channels: 1
  num_classes: 6
  class_names: [no_defect, porosity, crack, lack_of_fusion, undercut, slag_inclusion]
  class_weights: [1.0, 1.4, 1.6, 1.5, 1.2, 1.3]
augmentation:
  enabled: false
  policy: albumentations
  ops: []
model:
  architecture: custom_cnn
  pretrained: false
  dropout: 0.3
  ensemble_strategy: softmax_avg
training:
  epochs: 1
  batch_size: 4
  num_workers: 0
  optimizer: adamw
  lr: 0.001
  weight_decay: 0.0001
  scheduler: cosine_warm
  warmup_epochs: 0
  early_stopping:
    enabled: false
    patience: 7
    metric: val_f1_macro
    mode: max
  mixed_precision: false
  gradient_clip: 1.0
loss:
  name: focal_loss
  gamma: 2.0
  alpha: [1.0, 1.4, 1.6, 1.5, 1.2, 1.3]
metrics:
  primary: f1_macro
  also: [accuracy]
  confusion_matrix: true
  per_class_report: true
artifacts:
  checkpoint_dir: {tmp_path}/ckpt
  export_dir: {tmp_path}/models
  export_formats: [pt]
  log_dir: {tmp_path}/logs
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    metrics = train(str(cfg_path))
    assert "f1_macro" in metrics
    assert "checkpoint" in metrics
    assert (tmp_path / "models").exists()
