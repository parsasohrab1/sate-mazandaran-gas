"""End-to-end test: generate → train → evaluate → predict → serve."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_e2e_pipeline(tmp_path):
    """Full pipeline runs without errors on tiny synthetic data."""
    from weld_defect.data.synthetic import generate_synthetic_dataset
    from weld_defect.evaluate import evaluate
    from weld_defect.models.factory import build_model, save_model
    from weld_defect.predict import WeldPredictor
    from weld_defect.train import train

    # 1. Generate synthetic data.
    syn = generate_synthetic_dataset(n_samples=24, out_dir=tmp_path / "syn", seed=42, size=32)
    assert (syn / "manifest.csv").exists()

    # 2. Train for 1 epoch.
    config_text = f"""
experiment:
  name: e2e-test
  seed: 42
  mlflow_tracking_uri: file:{tmp_path}/mlruns
  mlflow_experiment: e2e-test
  mlflow_model_name: e2e-test
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
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(config_text, encoding="utf-8")

    train_metrics = train(str(cfg_path))
    assert "checkpoint" in train_metrics
    ckpt_path = train_metrics["checkpoint"]
    assert Path(ckpt_path).exists()

    # 3. Predict on a single image.
    predictor = WeldPredictor(checkpoint_path=ckpt_path, architecture="custom_cnn", input_size=32)
    import os
    image_files = sorted(os.listdir(syn / "images"))
    img_path = syn / "images" / image_files[0]
    pred = predictor.predict(str(img_path), image_name=image_files[0])
    assert pred["predicted_class"] in ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
    assert pred["decision_support_only"] is True

    # 4. Evaluate.
    eval_config_text = f"""
model:
  checkpoint: {ckpt_path}
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
    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(eval_config_text, encoding="utf-8")

    eval_result = evaluate(str(eval_cfg_path))
    assert "metrics" in eval_result
    assert "acceptance" in eval_result


def test_e2e_scripts(tmp_path):
    """Scripts generate_synthetic_data + prepare_data work."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent

    # generate_synthetic_data.py
    r1 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "generate_synthetic_data.py"),
         "--n-samples", "12", "--out", str(tmp_path / "syn"), "--seed", "42", "--size", "32"],
        capture_output=True, text=True, cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src"), "PATH": ""},  # minimal env
    )
    # Allow non-zero exit if deps missing, but check files exist.
    assert (tmp_path / "syn" / "manifest.csv").exists()

    # prepare_data.py
    r2 = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "prepare_data.py"),
         "--raw", str(tmp_path / "syn"), "--processed", str(tmp_path / "proc"), "--input-size", "32"],
        capture_output=True, text=True, cwd=repo_root,
        env={"PYTHONPATH": str(repo_root / "src"), "PATH": ""},
    )
    assert (tmp_path / "proc" / "manifest.csv").exists()
    assert (tmp_path / "proc" / "train.csv").exists()
    assert (tmp_path / "proc" / "data_profile.json").exists()
