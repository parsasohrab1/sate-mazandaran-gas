"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is on the path for tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def sample_image_uint8() -> np.ndarray:
    """A 224×224 grayscale uint8 synthetic image."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 255, (224, 224), dtype=np.uint8)


@pytest.fixture
def sample_image_hwc(sample_image_uint8: np.ndarray) -> np.ndarray:
    """A 224×224×3 HWC uint8 image."""
    return np.stack([sample_image_uint8] * 3, axis=-1)


@pytest.fixture
def tmp_synthetic_dataset(tmp_path):
    """Generate a tiny synthetic dataset (24 samples) in tmp_path."""
    from weld_defect.data.synthetic import generate_synthetic_dataset
    out = generate_synthetic_dataset(n_samples=24, out_dir=tmp_path / "syn", seed=42, size=64)
    return out


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """Train a tiny CustomCNN for 1 epoch on synthetic data and return its checkpoint path."""
    from weld_defect.data.synthetic import generate_synthetic_image
    from weld_defect.models.factory import build_model, save_model
    import torch

    model = build_model("custom_cnn", num_classes=6, pretrained=False)
    # Save untrained — tests just need a loadable checkpoint.
    ckpt_path = tmp_path / "tiny_model.pt"
    save_model(model, ckpt_path)
    return ckpt_path


@pytest.fixture
def manifest_df():
    """A small manifest DataFrame for split tests."""
    import pandas as pd
    rows = []
    for seg in range(6):  # 6 groups
        for cls_idx, cls in enumerate(["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]):
            for _ in range(3):  # 3 samples per group per class
                rows.append({"image": f"img_{seg}_{cls_idx}_{_}.png", "label": cls, "label_idx": cls_idx, "group": f"seg_{seg:02d}"})
    return pd.DataFrame(rows)
