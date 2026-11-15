"""Synthetic radiograph generator for smoke-testing the pipeline.

Generates grayscale images that mimic weld radiographs with optional defect
overlays (porosity dots, crack lines, lack-of-fusion bands, undercut grooves,
slag inclusions). Used by scripts/generate_synthetic_data.py and the CI ML
workflow to validate the full pipeline without real data (SRS §2.1, BRS §7).
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..utils.io import ensure_dir
from ..utils.seed import seed_everything

DEFECT_TYPES = [
    "no_defect",
    "porosity",
    "crack",
    "lack_of_fusion",
    "undercut",
    "slag_inclusion",
]


def _base_weld(size: int = 224, seed: int | None = None) -> np.ndarray:
    """Generate a base weld radiograph (grayscale) with a horizontal weld bead."""
    rng = np.random.RandomState(seed)
    img = rng.randint(80, 140, (size, size), dtype=np.uint8)
    # Weld bead: a horizontal band of slightly different intensity.
    bead_y = size // 2
    bead_h = size // 6
    img[bead_y - bead_h // 2 : bead_y + bead_h // 2, :] = np.clip(
        img[bead_y - bead_h // 2 : bead_y + bead_h // 2, :] + 30, 0, 255
    )
    # Add a Gaussian blur to smooth.
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


def _add_porosity(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Add scattered dark dots (gas pores)."""
    n = rng.randint(5, 20)
    for _ in range(n):
        y = rng.randint(0, img.shape[0])
        x = rng.randint(0, img.shape[1])
        r = rng.randint(2, 5)
        cv2.circle(img, (x, y), r, int(rng.randint(20, 50)), -1)
    return img


def _add_crack(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Add a thin dark line (crack)."""
    start = (rng.randint(0, img.shape[1]), rng.randint(0, img.shape[0]))
    end = (rng.randint(0, img.shape[1]), rng.randint(0, img.shape[0]))
    cv2.line(img, start, end, int(rng.randint(10, 30)), 1)
    return img


def _add_lack_of_fusion(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Add a straight dark band at the bead edge (lack of fusion)."""
    y = img.shape[0] // 2 + rng.randint(-10, 10)
    img[y : y + 2, :] = rng.randint(20, 40)
    return img


def _add_undercut(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Add a continuous groove at the bead edge (undercut)."""
    y = img.shape[0] // 2 - rng.randint(8, 15)
    img[y : y + 3, :] = np.clip(img[y : y + 3, :].astype(int) - 40, 0, 255).astype(np.uint8)
    return img


def _add_slag_inclusion(img: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Add an irregular dark blob (slag inclusion)."""
    y = rng.randint(0, img.shape[0])
    x = rng.randint(0, img.shape[1])
    w = rng.randint(8, 20)
    h = rng.randint(4, 10)
    cv2.ellipse(img, (x, y), (w, h), rng.randint(0, 180), 0, 360, int(rng.randint(15, 40)), -1)
    return img


_DEFECT_FNS = {
    "porosity": _add_porosity,
    "crack": _add_crack,
    "lack_of_fusion": _add_lack_of_fusion,
    "undercut": _add_undercut,
    "slag_inclusion": _add_slag_inclusion,
}


def generate_synthetic_image(
    size: int = 224,
    defect_type: str = "random",
    seed: int | None = None,
) -> np.ndarray:
    """Generate one synthetic radiograph.

    Args:
        size: Output image size (size × size).
        defect_type: One of DEFECT_TYPES or "random".
        seed: RNG seed for reproducibility.

    Returns:
        Grayscale uint8 array of shape (size, size).
    """
    rng = np.random.RandomState(seed)
    if defect_type == "random":
        defect_type = rng.choice(DEFECT_TYPES)

    if defect_type not in DEFECT_TYPES:
        raise ValueError(f"Unknown defect type: {defect_type}. Expected one of {DEFECT_TYPES}")

    img = _base_weld(size, seed)
    if defect_type != "no_defect":
        fn = _DEFECT_FNS[defect_type]
        img = fn(img, rng)
    return img


def generate_synthetic_dataset(
    n_samples: int = 1200,
    out_dir: str | Path = "data/synthetic",
    seed: int = 42,
    size: int = 224,
    test_size: float = 0.15,
    val_size: float = 0.10,
) -> Path:
    """Generate a full synthetic dataset with images + manifest + splits.

    Args:
        n_samples: Total number of images to generate.
        out_dir: Output directory.
        seed: RNG seed.
        size: Image size.
        test_size: Fraction for test split.
        val_size: Fraction for val split.

    Returns:
        Path to the output directory.
    """
    seed_everything(seed)
    out = ensure_dir(out_dir)
    images_dir = ensure_dir(out / "images")

    # Stratify: equal samples per class.
    n_per_class = n_samples // len(DEFECT_TYPES)
    records = []
    idx = 0
    for class_idx, class_name in enumerate(DEFECT_TYPES):
        for _ in range(n_per_class):
            img = generate_synthetic_image(size, class_name, seed=seed + idx)
            fname = f"sample_{idx:05d}.png"
            cv2.imwrite(str(images_dir / fname), img)
            records.append({
                "image": fname,
                "label": class_name,
                "label_idx": class_idx,
                "group": f"seg_{idx % 12:02d}",  # 12 pipeline segments for group split.
            })
            idx += 1

    df = pd.DataFrame(records)
    df.to_csv(out / "manifest.csv", index=False)

    # Stratified-group split.
    from .loader import stratified_group_split
    train_df, val_df, test_df = stratified_group_split(
        df, test_size=test_size, val_size=val_size, group_col="group", label_col="label", seed=seed,
    )
    train_df.to_csv(out / "train.csv", index=False)
    val_df.to_csv(out / "val.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)

    # Data profile.
    from .loader import compute_data_profile
    profile = compute_data_profile(df)
    from ..utils.io import save_json
    save_json(profile, out / "data_profile.json")

    return out
