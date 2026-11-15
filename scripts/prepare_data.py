#!/usr/bin/env python3
"""Prepare raw radiographs for training: preprocess + build splits.

Takes a directory of raw images (DICOM/PNG/JPG) plus an optional manifest CSV
(``image,label,group``) and produces:

    <processed>/images/*.png      — preprocessed (denoise+CLAHE+resize) PNGs
    <processed>/train.csv         — stratified-group train split
    <processed>/val.csv
    <processed>/test.csv
    <processed>/data_profile.json — basic data quality stats (for OPP/QPM)

If no manifest is supplied, labels are inferred from sub-directory names
(one class per folder) — a convenience for the synthetic and pilot phases.

Usage:
    python scripts/prepare_data.py \
        --raw data/raw/pilot --processed data/processed \
        --manifest data/raw/pilot/manifest.csv --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from weld_defect.data.preprocess import PreprocessPipeline
from weld_defect.data.loader import stratified_group_split
from weld_defect.utils import ensure_dir, get_logger

log = get_logger("prepare_data")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")


def _manifest_from_dirs(raw: Path) -> pd.DataFrame:
    rows = []
    for cls_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in EXTS:
                rows.append({"image": f.name, "label": cls_dir.name, "group": cls_dir.name})
    return pd.DataFrame(rows)


def _load_manifest(path: Path | None, raw: Path) -> pd.DataFrame:
    if path and path.exists():
        df = pd.read_csv(path)
        if "group" not in df.columns:
            df["group"] = "default"
        return df
    return _manifest_from_dirs(raw)


def profile(df: pd.DataFrame, processed: Path) -> dict:
    counts = df["label"].value_counts().to_dict()
    n_groups = df["group"].nunique()
    profile = {
        "n_samples": int(len(df)),
        "n_groups": int(n_groups),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": float(max(counts.values()) / max(1, min(counts.values()))),
        "missing_images": 0,
    }
    (processed / "data_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def main():
    p = argparse.ArgumentParser(description="Preprocess raw radiographs and build splits.")
    p.add_argument("--raw", type=Path, default=Path("data/raw/pilot"), help="raw image directory")
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    p.add_argument("--manifest", type=Path, default=None, help="CSV with image,label[,group]")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.15)
    args = p.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"raw directory not found: {args.raw}")

    processed = ensure_dir(args.processed)
    img_dir = ensure_dir(processed / "images")
    df = _load_manifest(args.manifest, args.raw)
    if df.empty:
        raise SystemExit(f"no images found under {args.raw}")

    pipe = PreprocessPipeline(input_size=args.input_size)
    n_written, missing = 0, 0
    for _, row in df.iterrows():
        src = args.raw / str(row["image"])
        if not src.exists():
            # try inside a class subdir
            src = args.raw / str(row.get("label", "")) / str(row["image"])
        if not src.exists():
            missing += 1
            continue
        arr = pipe(src)
        u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.min() < 0 else (arr * 255).astype(np.uint8)
        if u8.ndim == 3:
            u8 = u8[..., 0] if np.allclose(u8[..., 0], u8[..., -1]) else u8
        Image.fromarray(u8).save(img_dir / f"{Path(row['image']).stem}.png")
        n_written += 1
    log.info("preprocessed %d images (%d missing) → %s", n_written, missing, img_dir)

    # update manifest image column to processed stems
    df = df.copy()
    df["image"] = df["image"].apply(lambda s: f"{Path(str(s)).stem}.png")
    df = df[df["image"].apply(lambda s: (img_dir / s).exists())].reset_index(drop=True)

    tr, va, te = stratified_group_split(df, label_col="label", group_col="group",
                                        test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    tr.to_csv(processed / "train.csv", index=False)
    va.to_csv(processed / "val.csv", index=False)
    te.to_csv(processed / "test.csv", index=False)
    prof = profile(df, processed)
    log.info("splits → train=%d val=%d test=%d (imbalance=%.2f)", len(tr), len(va), len(te), prof["imbalance_ratio"])
    print(json.dumps({"processed": str(processed), "written": n_written, "missing": missing,
                      "train": len(tr), "val": len(va), "test": len(te), "profile": prof}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
