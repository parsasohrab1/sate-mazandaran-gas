"""Tests for data.preprocess."""

from __future__ import annotations

import numpy as np
import pytest

from weld_defect.data.preprocess import PreprocessPipeline, preprocess_image


def test_pipeline_grayscale_input(sample_image_uint8):
    """Pipeline processes a 2D grayscale array."""
    pipe = PreprocessPipeline(input_size=64)
    out = pipe(sample_image_uint8)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_pipeline_hwc_input(sample_image_hwc):
    """Pipeline processes a 3-channel HWC array."""
    pipe = PreprocessPipeline(input_size=64)
    out = pipe(sample_image_hwc)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32


def test_pipeline_different_input_size():
    """Pipeline resizes to the configured input_size."""
    img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
    pipe = PreprocessPipeline(input_size=128)
    out = pipe(img)
    assert out.shape == (128, 128, 3)


def test_pipeline_empty_raises():
    """Empty image raises ValueError."""
    pipe = PreprocessPipeline()
    with pytest.raises(ValueError):
        pipe(np.array([]))


def test_pipeline_invalid_shape_raises():
    """Invalid image shape raises ValueError."""
    pipe = PreprocessPipeline()
    with pytest.raises(ValueError):
        pipe(np.zeros((10, 10, 10), dtype=np.uint8))


def test_preprocess_image_from_file(tmp_path, sample_image_uint8):
    """preprocess_image loads and processes a PNG file."""
    import cv2
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), sample_image_uint8)
    out = preprocess_image(str(p), input_size=64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.float32


def test_preprocess_image_missing_raises(tmp_path):
    """Missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_image(str(tmp_path / "nope.png"))


def test_pipeline_reproducible():
    """Same input → same output (deterministic)."""
    img = np.random.RandomState(0).randint(0, 255, (64, 64), dtype=np.uint8)
    pipe = PreprocessPipeline(input_size=32)
    out1 = pipe(img)
    out2 = pipe(img)
    np.testing.assert_array_equal(out1, out2)
