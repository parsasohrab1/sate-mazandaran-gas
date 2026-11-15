"""Tests for data.synthetic."""

from __future__ import annotations

import numpy as np
import pytest

from weld_defect.data.synthetic import (
    DEFECT_TYPES,
    generate_synthetic_dataset,
    generate_synthetic_image,
)


def test_generate_synthetic_image_shape():
    """Image has the requested size."""
    img = generate_synthetic_image(size=128, defect_type="no_defect", seed=0)
    assert img.shape == (128, 128)
    assert img.dtype == np.uint8


def test_generate_synthetic_image_all_defects():
    """All defect types produce valid images."""
    for dt in DEFECT_TYPES:
        img = generate_synthetic_image(size=64, defect_type=dt, seed=1)
        assert img.shape == (64, 64)
        assert img.dtype == np.uint8


def test_generate_synthetic_image_random():
    """Random defect type works."""
    img = generate_synthetic_image(size=64, defect_type="random", seed=2)
    assert img.shape == (64, 64)


def test_generate_synthetic_image_invalid_defect():
    """Invalid defect type raises ValueError."""
    with pytest.raises(ValueError):
        generate_synthetic_image(defect_type="not_a_defect")


def test_generate_synthetic_image_reproducible():
    """Same seed → same image."""
    img1 = generate_synthetic_image(size=64, seed=42)
    img2 = generate_synthetic_image(size=64, seed=42)
    np.testing.assert_array_equal(img1, img2)


def test_generate_synthetic_dataset(tmp_path):
    """Dataset generation produces expected files."""
    out = generate_synthetic_dataset(
        n_samples=12,
        out_dir=tmp_path / "syn",
        seed=42,
        size=32,
    )
    assert (out / "manifest.csv").exists()
    assert (out / "train.csv").exists()
    assert (out / "val.csv").exists()
    assert (out / "test.csv").exists()
    assert (out / "data_profile.json").exists()
    assert (out / "images").is_dir()

    import pandas as pd
    manifest = pd.read_csv(out / "manifest.csv")
    assert len(manifest) == 12
    assert "image" in manifest.columns
    assert "label" in manifest.columns
    assert "label_idx" in manifest.columns
    assert "group" in manifest.columns


def test_generate_synthetic_dataset_splits_disjoint(tmp_path):
    """Train/val/test splits have no group leakage."""
    out = generate_synthetic_dataset(
        n_samples=24,
        out_dir=tmp_path / "syn",
        seed=42,
        size=32,
    )
    import pandas as pd
    train = pd.read_csv(out / "train.csv")
    val = pd.read_csv(out / "val.csv")
    test = pd.read_csv(out / "test.csv")
    assert set(train["group"]).isdisjoint(set(val["group"]))
    assert set(train["group"]).isdisjoint(set(test["group"]))
    assert set(val["group"]).isdisjoint(set(test["group"]))
