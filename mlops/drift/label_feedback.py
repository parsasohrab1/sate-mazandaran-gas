#!/usr/bin/env python3
"""Label feedback loop — reviewed samples feed retraining (OID, serving_policy §human_review).

Reads expert review records (approve/reject + corrected label) from the dashboard
export and appends them to a retraining queue. These samples are prioritized in
the next retrain run (active learning on edge cases).

Usage:
    python mlops/drift/label_feedback.py --reviews metrics/human_review.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def run(reviews_path: str, queue_path: str = "data/processed/relabel_queue.csv"):
    p = Path(reviews_path)
    if not p.exists():
        print(f"[feedback] no reviews at {p}; nothing to do.")
        return
    rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Keep only rejections + corrections (these are the high-signal samples)
    df = df[df.get("status", "").isin(["rejected", "corrected"]) | df.get("corrected_label").notna()]
    if df.empty:
        print("[feedback] no rejections/corrections; nothing to queue.")
        return
    out_cols = ["image", "corrected_label", "group", "reviewer", "reason"]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None
    df["corrected_label"] = df["corrected_label"].fillna(df.get("predicted_class"))
    out = Path(queue_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = not out.exists()
    df[out_cols].to_csv(out, mode="a", header=header, index=False)
    print(f"[feedback] queued {len(df)} reviewed samples for next retrain → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reviews", default="metrics/human_review.jsonl")
    p.add_argument("--queue", default="data/processed/relabel_queue.csv")
    args = p.parse_args()
    run(args.reviews, args.queue)


if __name__ == "__main__":
    main()
