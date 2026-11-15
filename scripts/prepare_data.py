#!/usr/bin/env python3
"""Prepare raw data for training: preprocess + split + profile.

Usage:
  python scripts/prepare_data.py --raw data/raw/<src> --processed data/processed
  python scripts/prepare_data.py --raw data/synthetic --processed data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weld_defect.data.loader import compute_data_profile, stratified_group_split
from weld_defect.data.preprocess import PreprocessPipeline
from weld_defect.utils.io import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare raw data for training")
    parser.add_argument("--raw", required=True, help="Raw data directory (must contain manifest.csv or images/)")
    parser.add_argument("--processed", default="data/processed", help="Output directory")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path(args.raw)
    processed = Path(args.processed)
    images_out = processed / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Load or build manifest.
    manifest_path = raw / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        # Build manifest from images/ directory; assume label encoded in filename.
        images_dir = raw / "images" if (raw / "images").exists() else raw
        if not images_dir.exists():
            print(f"Error: no manifest.csv and no images/ directory in {raw}", file=sys.stderr)
            return 1

        from weld_defect.config import CLASS_NAMES
        records = []
        for img_path in sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg")):
            stem = img_path.stem.lower()
            label = "no_defect"
            for cls in CLASS_NAMES:
                if cls in stem:
                    label = cls
                    break
            records.append({
                "image": img_path.name,
                "label": label,
                "label_idx": CLASS_NAMES.index(label),
                "group": f"seg_{hash(stem) % 12:02d}",
            })
        manifest = pd.DataFrame(records)

    if len(manifest) == 0:
        print(f"Error: empty manifest from {raw}", file=sys.stderr)
        return 1

    # Preprocess images.
    raw_images_dir = raw / "images" if (raw / "images").exists() else raw
    pipeline = PreprocessPipeline(input_size=args.input_size)
    import cv2
    import numpy as np

    processed_count = 0
    for _, row in manifest.iterrows():
        src = raw_images_dir / row["image"]
        if not src.exists():
            continue
        raw_bytes = np.fromfile(str(src), dtype=np.uint8)
        img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        processed_img = pipeline(img)
        # Save as uint8 PNG.
        out_img = (processed_img * 255).clip(0, 255).astype(np.uint8)
        out_path = images_out / row["image"]
        cv2.imwrite(str(out_path), out_img)
        processed_count += 1

    print(f"Preprocessed {processed_count}/{len(manifest)} images → {images_out}")

    # Split.
    if "group" not in manifest.columns:
        manifest["group"] = [f"seg_{i % 12:02d}" for i in range(len(manifest))]
    if "label_idx" not in manifest.columns:
        from weld_defect.config import CLASS_NAMES
        manifest["label_idx"] = manifest["label"].map({n: i for i, n in enumerate(CLASS_NAMES)}).fillna(0).astype(int)

    train_df, val_df, test_df = stratified_group_split(
        manifest, test_size=args.test_size, val_size=args.val_size,
        group_col="group", label_col="label", seed=args.seed,
    )
    train_df.to_csv(processed / "train.csv", index=False)
    val_df.to_csv(processed / "val.csv", index=False)
    test_df.to_csv(processed / "test.csv", index=False)
    manifest.to_csv(processed / "manifest.csv", index=False)

    # Profile.
    profile = compute_data_profile(manifest)
    profile["processed_images"] = processed_count
    save_json(profile, processed / "data_profile.json")

    print(f"Splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Profile: {profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
