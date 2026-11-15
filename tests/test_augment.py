"""Tests for data.augment."""

from __future__ import annotations

import numpy as np
import pytest

from weld_defect.data.augment import build_augmentation, get_train_aug, get_val_aug


def test_get_val_aug_returns_compose():
    """get_val_aug returns an Albumentations Compose."""
    transform = get_val_aug(64)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = transform(image=img)
    assert "image" in result
    assert result["image"].shape == (64, 64, 3)


def test_get_train_aug_empty_ops():
    """get_train_aug with empty ops falls back to val aug."""
    transform = get_train_aug([], 64)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (64, 64, 3)


def test_get_train_aug_with_ops():
    """get_train_aug with ops builds a pipeline."""
    ops = [
        {"name": "HorizontalFlip", "p": 0.5},
        {"name": "Rotate", "limit": 15, "p": 0.5},
    ]
    transform = get_train_aug(ops, 64)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (64, 64, 3)


def test_build_augmentation_unknown_op_raises():
    """build_augmentation raises for unknown ops."""
    with pytest.raises(ValueError):
        build_augmentation([{"name": "NonexistentOp", "p": 0.5}], 64)


def test_build_augmentation_clahe():
    """CLAHE op is mapped correctly (case-insensitive alias)."""
    ops = [{"name": "CLAHE", "clip_limit": 4.0, "p": 0.3}]
    transform = build_augmentation(ops, 64)
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = transform(image=img)
    assert result["image"].shape == (64, 64, 3)
