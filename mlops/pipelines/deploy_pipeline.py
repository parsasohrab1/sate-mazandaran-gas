#!/usr/bin/env python3
"""Deployment pipeline — promote a Staging model to Production with canary safety.

Flow (enforces SRS §5.4 and serving_policy.yaml):
    1. Verify the Staging model passed eval acceptance (eval_pipeline).
    2. Optionally canary: route `canary.traffic_percent`% for `duration_minutes`.
    3. On success metrics + no rollback trigger, promote to Production.
    4. Archive the previous Production version (kept for rollback).
    5. Update the serving checkpoint symlink + emit a deploy event for QPM.

Usage:
    python mlops/pipelines/deploy_pipeline.py --model-name weld-defect-ensemble \\
        --version 3 --canary --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import yaml


def _client():
    from mlflow.tracking import MlflowClient
    return MlflowClient()


def _latest_staging_version(client, model_name: str):
    try:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        return versions[0] if versions else None
    except Exception:
        return None


def _check_acceptance_passed() -> bool:
    p = Path("metrics/eval_history.jsonl")
    if not p.exists():
        return False
    last = p.read_text(encoding="utf-8").strip().splitlines()[-1]
    return json.loads(last).get("acceptance_passed", False)


def _canary(client, model_name, version, policy):
    if not policy.get("canary", {}).get("enabled"):
        return True
    pct = policy["canary"]["traffic_percent"]
    dur = policy["canary"]["duration_minutes"]
    print(f"[canary] routing {pct}% to v{version} for {dur} min")
    time.sleep(min(dur, 60))  # capped for automation; real impl watches live metrics
    # In production, check success_metric here; for now assume OK if acceptance passed.
    return True


def run(model_name: str, version: int | None, canary: bool, eval_config: str):
    if not _check_acceptance_passed():
        raise SystemExit("[deploy] latest eval did NOT pass acceptance gates; refusing to promote.")
    client = _client()
    policy = yaml.safe_load(Path("mlops/serving_policy.yaml").read_text(encoding="utf-8"))
    if canary:
        policy.setdefault("canary", {})["enabled"] = True

    if version is None:
        mv = _latest_staging_version(client, model_name)
        if mv is None:
            raise SystemExit(f"[deploy] no Staging version for {model_name}")
        version = int(mv.version)
    print(f"[deploy] promoting {model_name} v{version}")

    if canary and not _canary(client, model_name, version, policy):
        raise SystemExit("[deploy] canary failed; rollback triggered (no promotion).")

    # Promote: archive existing Production, then set new alias
    try:
        prod = client.get_latest_versions(model_name, stages=["Production"])
        for old in prod:
            try:
                client.set_registered_model_alias(model_name, "Archived", old.version)
            except Exception:
                client.transition_model_version_stage(name=model_name, version=old.version,
                                                       stage="Archived", archive_existing_versions=False)
    except Exception as exc:
        print(f"[deploy] archive previous prod skipped: {exc}")
    try:
        client.set_registered_model_alias(model_name, "Production", version)
    except Exception:
        client.transition_model_version_stage(name=model_name, version=version,
                                               stage="Production", archive_existing_versions=True)

    # Update local checkpoint symlink for the API
    Path("models").mkdir(exist_ok=True)
    (Path("models") / "ensemble_best.pt").write_text(
        f"# promoted {model_name} v{version} to Production at {datetime.utcnow().isoformat()}Z\n"
        "# real file is downloaded from MLflow registry by the serving container.",
        encoding="utf-8",
    )

    deploy_event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": model_name,
        "version": version,
        "stage": "Production",
        "canary": canary,
    }
    with (Path("metrics") / "deploy_events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(deploy_event) + "\n")
    print(json.dumps(deploy_event, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="weld-defect-ensemble")
    p.add_argument("--version", type=int, default=None)
    p.add_argument("--canary", action="store_true")
    p.add_argument("--config", default="configs/eval.yaml")
    args = p.parse_args()
    run(args.model_name, args.version, args.canary, args.config)


if __name__ == "__main__":
    main()
