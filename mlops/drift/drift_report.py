#!/usr/bin/env python3
"""Data/concept drift detection (OPP feed, QPM §2 trigger).

Compares the feature/label distribution of the current serving window against a
reference (the training-set profile). Uses Evidently if installed; otherwise
falls back to a pure-numpy Population Stability Index (PSI) on per-class
prediction frequencies and image-statistics histograms.

Outputs:
    metrics/drift/drift_report.json   — PSI per feature + overall verdict
    metrics/drift/drift_report.html   — (only if Evidently available)

Usage:
    python mlops/drift/drift_report.py \
        --reference metrics/baselines/reference_profile.json \
        --current   metrics/latest/current_profile.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions."""
    eps = 1e-6
    edges = np.linspace(0, 1, bins + 1)  # works for normalized [0,1] features
    exp_hist, _ = np.histogram(expected, bins=edges)
    act_hist, _ = np.histogram(actual, bins=edges)
    exp_pct = (exp_hist + eps) / (exp_hist.sum() + eps * bins)
    act_pct = (act_hist + eps) / (act_hist.sum() + eps * bins)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _profile_from_images(image_dir: Path, n: int = 200) -> dict:
    """Cheap image-statistic profile: mean, std, fraction-dark per image."""
    files = sorted(image_dir.glob("*.png"))[:n]
    means, stds, darks = [], [], []
    from PIL import Image
    for f in files:
        a = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
        means.append(float(a.mean()))
        stds.append(float(a.std()))
        darks.append(float((a < 0.3).mean()))
    return {"n": len(files), "means": means, "stds": stds, "darks": darks}


def compute_drift(reference: dict, current: dict) -> dict:
    """Return per-feature PSI and an overall verdict."""
    feats = {}
    for key in ("means", "stds", "darks"):
        if key in reference and key in current:
            ref = np.asarray(reference[key], dtype=np.float32)
            cur = np.asarray(current[key], dtype=np.float32)
            feats[f"img_{key}_psi"] = _psi(np.clip(ref, 0, 1), np.clip(cur, 0, 1))
    # class distribution drift (prediction frequencies)
    ref_classes = reference.get("class_counts") or reference.get("pred_counts") or {}
    cur_classes = current.get("class_counts") or current.get("pred_counts") or {}
    if ref_classes and cur_classes:
        names = sorted(set(ref_classes) | set(cur_classes))
        ref_arr = np.array([ref_classes.get(k, 0) for k in names], dtype=float)
        cur_arr = np.array([cur_classes.get(k, 0) for k in names], dtype=float)
        ref_arr = ref_arr / ref_arr.sum()
        cur_arr = cur_arr / cur_arr.sum()
        feats["class_dist_psi"] = float(np.sum((cur_arr - ref_arr) * np.log((cur_arr + 1e-9) / (ref_arr + 1e-9))))
    overall = max(feats.values()) if feats else 0.0
    verdict = "ok" if overall < 0.1 else ("watch" if overall < 0.25 else "alarm")
    return {"features": feats, "max_psi": overall, "verdict": verdict,
            "thresholds": {"ok": 0.1, "watch": 0.25}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="reference profile JSON (training)")
    p.add_argument("--current", required=True, help="current profile JSON (serving window)")
    p.add_argument("--out", default="metrics/drift/drift_report.json")
    args = p.parse_args()

    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    cur = json.loads(Path(args.current).read_text(encoding="utf-8"))
    report = compute_drift(ref, cur)

    # Optional: Evidently HTML report
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        import pandas as pd
        ref_df = pd.DataFrame({k: ref.get(k, []) for k in ("means", "stds", "darks")})
        cur_df = pd.DataFrame({k: cur.get(k, []) for k in ("means", "stds", "darks")})
        r = Report(metrics=[DataDriftPreset()])
        r.run(reference_data=ref_df, current_data=cur_df)
        html_path = Path(args.out).with_suffix(".html")
        r.save_html(str(html_path))
        report["html_report"] = str(html_path)
    except Exception:
        pass

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["verdict"] == "alarm":
        print("[drift] ALARM — QPM §5 / CAR §2 trigger raised.")


if __name__ == "__main__":
    main()
