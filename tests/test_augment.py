"""Augmenter tests — numpy fallback path (no albumentations required)."""
from __future__ import annotations

import numpy as np

from weld_defect.data.augment import Augmenter, build_augmenter
from weld_defect.config import AugmentConfig


def test_augmenter_no_ops_passes_through():
    aug = Augmenter(ops=[], input_size=32)
    img = np.zeros((32, 32), dtype=np.float32)
    out = aug(img)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32


def test_augmenter_2d_input_becomes_3channel():
    aug = Augmenter(ops=[], input_size=24)
    out = aug(np.full((40, 40), 0.5, dtype=np.float32))
    assert out.shape == (24, 24, 3)


def test_augmenter_3d_single_channel():
    aug = Augmenter(ops=[], input_size=16)
    out = aug(np.full((20, 20, 1), 0.5, dtype=np.float32))
    assert out.shape == (16, 16, 3)


def test_build_augmenter_from_disabled_config():
    cfg = AugmentConfig(enabled=False)
    aug = build_augmenter(cfg)
    # build_augmenter defaults to input_size=224
    img = np.zeros((32, 32), dtype=np.float32)
    out = aug(img)
    assert out.shape == (224, 224, 3)


def test_build_augmenter_from_enabled_config():
    cfg = AugmentConfig(enabled=True, ops=[{"name": "HorizontalFlip", "p": 0.5}])
    aug = build_augmenter(cfg)
    assert aug.input_size == 224
