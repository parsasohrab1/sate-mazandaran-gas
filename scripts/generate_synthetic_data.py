#!/usr/bin/env python3
"""Synthetic radiograph generator for first validation (CMMI ML5 — QPM/OPP).

Why synthetic first?
    Real pipeline radiographs from Sate Mazandaran Gas are governed by a data
    sharing agreement (see docs/DataDictionary.md §3) and arrive after the
    labeling protocol is signed off. To de-risk the ML pipeline, the toolchain
    (preprocess → augment → train → evaluate → serve) is validated end-to-end
    on synthetic radiographs that mimic the visual statistics of real X-ray
    weld images: a bright horizontal weld bead on a darker base-metal
    background, with the six API-1104-style defect classes injected
    deterministically.

Classes generated (matches configs/train.yaml → data.class_names):
    0  no_defect       — clean weld bead
    1  porosity        — round dark gas cavities inside the bead
    2  crack           — thin dark longitudinal/transverse line
    3  lack_of_fusion  — dark straight band along the bead edge
    4  undercut        — continuous dark groove at the bead toe
    5  slag_inclusion  — irregular elongated dark inclusion

Outputs:
    <out>/images/<group>__<idx>.png   — 256x256 8-bit grayscale PNG
    <out>/manifest.csv               — columns: image,label,label_idx,group
    <out>/train.csv, val.csv, test.csv — stratified-group splits

Usage:
    python scripts/generate_synthetic_data.py --n-samples 1200 --out data/synthetic
    python scripts/generate_synthetic_data.py --n-samples 60 --size 64 --out data/synthetic_small
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]


def _base_radiograph(size: int, rng: np.random.Generator) -> np.ndarray:
    """Background: dark base metal + brighter horizontal weld bead (centered)."""
    img = rng.normal(loc=0.35, scale=0.04, size=(size, size)).clip(0, 1)
    # weld bead: a Gaussian ridge along the horizontal midline
    yy = np.arange(size).reshape(-1, 1)
    bead_center = size // 2 + rng.integers(-size // 16, size // 16 + 1)
    sigma = max(3.0, size / 18.0)
    ridge = np.exp(-((yy - bead_center) ** 2) / (2 * sigma ** 2)) * 0.35
    img = np.clip(img + ridge, 0, 1)
    # subtle vertical texture (weld ripples)
    ripple = np.sin(np.linspace(0, 8 * np.pi, size))[None, :] * 0.02
    img = np.clip(img + ripple, 0, 1)
    # add film grain noise
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)
    return img.astype(np.float32), bead_center, sigma


def _blob(img, cx, cy, r, darkness, rng):
    yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    img[mask] = np.clip(img[mask] - darkness, 0, 1)


def _inject_defect(img, label_idx, bead_center, sigma, rng) -> np.ndarray:
    size = img.shape[0]
    if label_idx == 0:  # no_defect
        return img
    if label_idx == 1:  # porosity
        n = rng.integers(3, 8)
        for _ in range(n):
            cx = rng.integers(int(size * 0.1), int(size * 0.9))
            cy = bead_center + rng.integers(-int(sigma), int(sigma) + 1)
            r = rng.integers(2, max(3, int(sigma * 0.4)))
            _blob(img, cx, cy, r, darkness=rng.uniform(0.25, 0.45), rng=rng)
    elif label_idx == 2:  # crack
        horizontal = rng.random() < 0.5
        cx = rng.integers(int(size * 0.15), int(size * 0.85))
        length = rng.integers(int(size * 0.15), int(size * 0.4))
        thickness = rng.integers(1, 3)
        if horizontal:
            y0 = bead_center + rng.integers(-int(sigma), int(sigma) + 1)
            img[y0 - thickness // 2: y0 + thickness // 2 + 1, cx: cx + length] = np.clip(
                img[y0 - thickness // 2: y0 + thickness // 2 + 1, cx: cx + length] - 0.5, 0, 1)
        else:
            x0 = cx
            y0 = rng.integers(int(size * 0.1), int(size * 0.9) - length)
            img[y0: y0 + length, x0 - thickness // 2: x0 + thickness // 2 + 1] = np.clip(
                img[y0: y0 + length, x0 - thickness // 2: x0 + thickness // 2 + 1] - 0.5, 0, 1)
    elif label_idx == 3:  # lack_of_fusion: dark straight band along one bead edge
        side = 1 if rng.random() < 0.5 else -1
        edge_y = bead_center + int(side * sigma * 0.9)
        w = rng.integers(2, 4)
        img[edge_y - w: edge_y + w, :] = np.clip(img[edge_y - w: edge_y + w, :] - 0.35, 0, 1)
    elif label_idx == 4:  # undercut: continuous dark groove at bead toe
        side = 1 if rng.random() < 0.5 else -1
        edge_y = bead_center + int(side * sigma * 1.1)
        w = 2
        img[edge_y: edge_y + w, :] = np.clip(img[edge_y: edge_y + w, :] - 0.4, 0, 1)
    elif label_idx == 5:  # slag_inclusion: irregular elongated dark blob
        cx = rng.integers(int(size * 0.2), int(size * 0.8))
        cy = bead_center + rng.integers(-int(sigma * 0.5), int(sigma * 0.5) + 1)
        rw = rng.integers(6, 14)
        rh = rng.integers(3, 6)
        yy, xx = np.ogrid[:size, :size]
        mask = (((yy - cy) / rh) ** 2 + ((xx - cx) / rw) ** 2) <= 1
        img[mask] = np.clip(img[mask] - rng.uniform(0.3, 0.45), 0, 1)
    return img


def generate(n_samples: int, size: int, seed: int, out: Path, n_groups: int = 12):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # balance classes (slight majority for no_defect to mimic real prevalence)
    weights = [0.30, 0.14, 0.14, 0.14, 0.14, 0.14]
    labels = py_rng.choices(range(len(CLASS_NAMES)), weights=weights, k=n_samples)
    groups = [f"G{g:02d}" for g in py_rng.choices(range(n_groups), k=n_samples)]

    rows = []
    for i, (li, g) in enumerate(zip(labels, groups)):
        base, bc, sigma = _base_radiograph(size, rng)
        img = _inject_defect(base.copy(), li, bc, sigma, rng)
        fname = f"{g}__{i:05d}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(img_dir / fname)
        rows.append({"image": fname, "label": CLASS_NAMES[li], "label_idx": li, "group": g})

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    # stratified group split via the package's own splitter
    from weld_defect.data.loader import stratified_group_split

    tr, va, te = stratified_group_split(df, label_col="label_idx", group_col="group",
                                        test_size=0.2, val_size=0.15, seed=seed)
    tr.to_csv(out / "train.csv", index=False)
    va.to_csv(out / "val.csv", index=False)
    te.to_csv(out / "test.csv", index=False)
    print(f"Generated {n_samples} synthetic radiographs ({size}x{size}) → {out}")
    print(f"  train={len(tr)} val={len(va)} test={len(te)}  classes={CLASS_NAMES}")
    print(f"  manifest → {out / 'manifest.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--size", type=int, default=256, help="image edge length in pixels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = p.parse_args()
    generate(args.n_samples, args.size, args.seed, args.out)


if __name__ == "__main__":
    main()
