#!/usr/bin/env python3
"""Evaluation + acceptance-gate pipeline.

Runs weld_defect.evaluate, then enforces the BRS acceptance gates. If the
Staging model passes, it is eligible for promotion (handled by deploy_pipeline).
If it fails, a CAR entry is seeded.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml


def run(config: str):
    from weld_defect.evaluate import evaluate as _eval
    out = _eval(config)
    metrics = out["metrics"]
    acceptance = out["acceptance"]

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "f1_macro": metrics["f1_macro"],
        "recall_crack": metrics["per_class"].get("crack", {}).get("recall"),
        "precision_no_defect": metrics["per_class"].get("no_defect", {}).get("precision"),
        "inference_ms_mean": out["inference_ms_mean"],
        "acceptance_passed": acceptance["passed"],
        "gates": acceptance["checks"],
    }

    # Append to QPM history (OPP feed)
    Path("metrics").mkdir(exist_ok=True)
    hist = Path("metrics/eval_history.jsonl")
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # Seed a CAR entry on failure (CAR §2 trigger)
    if not acceptance["passed"]:
        car_log = Path("metrics/car_log.yaml")
        existing = yaml.safe_load(car_log.read_text(encoding="utf-8")) if car_log.exists() else []
        existing = existing or []
        existing.append({
            "id": f"CAR-AUTO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "date": datetime.utcnow().date().isoformat(),
            "trigger": "acceptance gate failure in eval_pipeline",
            "failed_gates": [g for g in acceptance["checks"] if not g.get("passed")],
            "status": "open",
            "scope": "staging candidate",
        })
        car_log.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print("[CAR] acceptance failed — seeded a CAR entry. See metrics/car_log.yaml")
    else:
        print("[eval] acceptance PASSED — eligible for promotion.")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
