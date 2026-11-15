#!/usr/bin/env python3
"""Generate a synthetic weld radiograph dataset for smoke-testing the pipeline.

Usage:
  python scripts/generate_synthetic_data.py --n-samples 1200 --out data/synthetic --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/generate_synthetic_data.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from weld_defect.data.synthetic import generate_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic weld radiograph dataset")
    parser.add_argument("--n-samples", type=int, default=1200, help="Total number of images")
    parser.add_argument("--out", default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--size", type=int, default=224, help="Image size (size × size)")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split fraction")
    parser.add_argument("--val-size", type=float, default=0.10, help="Val split fraction")
    args = parser.parse_args()

    print(f"Generating {args.n_samples} synthetic images → {args.out}")
    out_dir = generate_synthetic_dataset(
        n_samples=args.n_samples,
        out_dir=args.out,
        seed=args.seed,
        size=args.size,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    print(f"Done. Dataset written to {out_dir}")
    print(f"  - {out_dir}/manifest.csv")
    print(f"  - {out_dir}/train.csv, val.csv, test.csv")
    print(f"  - {out_dir}/data_profile.json")
    print(f"  - {out_dir}/images/*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
