"""Synthetic data generator tests — numpy/Pillow/pandas/sklearn (no torch)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import sys
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_synthetic_data import generate, CLASS_NAMES, _base_radiograph, _inject_defect


def test_base_radiograph_shape():
    rng = np.random.default_rng(0)
    img, bc, sigma = _base_radiograph(64, rng)
    assert img.shape == (64, 64)
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert 0 <= bc < 64
    assert sigma > 0


def test_inject_each_defect_class():
    rng = np.random.default_rng(1)
    for label_idx in range(6):
        base, bc, sigma = _base_radiograph(64, rng)
        out = _inject_defect(base.copy(), label_idx, bc, sigma, rng)
        assert out.shape == (64, 64)


def test_generate_creates_splits(tmp_path):
    out = tmp_path / "syn"
    generate(n_samples=60, size=48, seed=42, out=out, n_groups=6)
    # images written
    assert (out / "images").is_dir()
    imgs = list((out / "images").glob("*.png"))
    assert len(imgs) == 60
    # manifest + splits
    assert (out / "manifest.csv").exists()
    assert (out / "train.csv").exists()
    assert (out / "val.csv").exists()
    assert (out / "test.csv").exists()
    df = pd.read_csv(out / "manifest.csv")
    assert set(df.columns) >= {"image", "label", "label_idx", "group"}
    assert set(df["label"].unique()).issubset(set(CLASS_NAMES))
    # splits sum to total
    tr = pd.read_csv(out / "train.csv")
    va = pd.read_csv(out / "val.csv")
    te = pd.read_csv(out / "test.csv")
    assert len(tr) + len(va) + len(te) == 60
    # no group leakage
    assert set(tr["group"]).isdisjoint(set(va["group"]))
    assert set(tr["group"]).isdisjoint(set(te["group"]))
