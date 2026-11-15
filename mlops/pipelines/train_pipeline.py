#!/usr/bin/env python3
"""End-to-end training pipeline (MLOps).

Wraps weld_defect.train with:
  - DVC data pull (optional)
  - resolved config snapshot to MLflow
  - promotion of the resulting checkpoint to the MLflow registry as 'Staging'
  - emission of a QPM/OPP metrics summary

Usage:
    python mlops/pipelines/train_pipeline.py --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def _dvc_pull():
    if os.environ.get("MLOPS_SKIP_DVC"):
        return
    try:
        import subprocess
        subprocess.run(["dvc", "pull"], check=False)
    except FileNotFoundError:
        pass


def _promote_to_registry(checkpoint: str, model_name: str, tags: dict):
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        # Register the artifact
        mlflow.log_artifact(checkpoint, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        uri = f"runs:/{run_id}/model"
        mv = client.create_model_version(name=model_name, source=uri, run_id=run_id, tags=tags)
        try:
            client.set_registered_model_alias(model_name, "Staging", mv.version)
        except Exception:
            client.transition_model_version_stage(name=model_name, version=mv.version,
                                                   stage="Staging", archive_existing_versions=False)
        print(f"[registry] {model_name} v{mv.version} → Staging")
    except Exception as exc:
        print(f"[registry] promotion skipped: {exc}")


def run(config: str):
    _dvc_pull()
    from weld_defect.train import train as _train
    out = _train(config)
    # Log resolved config + QPM summary
    cfg_yaml = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    try:
        import mlflow
        mlflow.log_param("resolved_config", json.dumps(cfg_yaml, ensure_ascii=False)[:4000])
        mlflow.set_tags({"pipeline": "train", "cmmi_level": "5"})
    except Exception:
        pass
    model_name = cfg_yaml.get("experiment", {}).get("mlflow_model_name", "weld-defect-ensemble")
    _promote_to_registry(out["checkpoint"], model_name,
                         tags={"source": "train_pipeline", "val_f1_macro": str(out["best_val_f1_macro"])})
    print(json.dumps({"checkpoint": out["checkpoint"], "best_val_f1_macro": out["best_val_f1_macro"]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train.yaml")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
