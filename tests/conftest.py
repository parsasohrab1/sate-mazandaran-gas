"""Shared pytest fixtures and path setup.

Makes ``weld_defect`` importable without an editable install (adds ``src`` to
``sys.path``), and provides small synthetic-image fixtures so unit tests do not
depend on real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("weld_data")


@pytest.fixture
def gray_image_array():
    """A 64x64 float32 radiograph-like array in [0,1]."""
    rng = np.random.default_rng(0)
    img = rng.normal(0.4, 0.05, (64, 64)).clip(0, 1).astype(np.float32)
    yy = np.arange(64).reshape(-1, 1)
    img = np.clip(img + np.exp(-((yy - 32) ** 2) / 200) * 0.3, 0, 1)
    return img


@pytest.fixture
def png_image(tmp_path, gray_image_array):
    p = tmp_path / "sample.png"
    Image.fromarray((gray_image_array * 255).astype(np.uint8)).save(p)
    return p


@pytest.fixture
def small_manifest_df():
    """A toy manifest with 6 groups × 6 classes for split tests."""
    rows = []
    for g in range(6):
        for c in range(6):
            for i in range(4):
                rows.append({"image": f"g{g}_c{c}_{i}.png",
                             "label": ["no_defect","porosity","crack","lack_of_fusion","undercut","slag_inclusion"][c],
                             "label_idx": c, "group": f"G{g:02d}"})
    return pd.DataFrame(rows)


@pytest.fixture
def class_names():
    return ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
