"""Preprocess pipeline tests — numpy/Pillow only (no torch)."""
from __future__ import annotations

import numpy as np

from weld_defect.data.preprocess import (
    PreprocessPipeline, load_image_gray, denoise, clahe, resize_to,
    normalize, replicate_channels, to_tensor, self_check_single,
)


def test_load_image_gray(png_image):
    arr = load_image_gray(png_image)
    assert arr.ndim == 2
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_denoise_preserves_shape(gray_image_array):
    out = denoise(gray_image_array)
    assert out.shape == gray_image_array.shape


def test_clahe_preserves_shape(gray_image_array):
    out = clahe(gray_image_array)
    assert out.shape == gray_image_array.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_resize_to():
    arr = np.zeros((40, 50), dtype=np.float32)
    out = resize_to(arr, size=32)
    assert out.shape == (32, 32)


def test_normalize_centers():
    arr = np.full((8, 8), 0.5, dtype=np.float32)
    out = normalize(arr, mean=0.5, std=0.5)
    assert np.allclose(out, 0.0)


def test_replicate_channels():
    arr = np.zeros((4, 4), dtype=np.float32)
    out = replicate_channels(arr, 3)
    assert out.shape == (4, 4, 3)
    assert np.allclose(out[..., 0], out[..., 2])


def test_pipeline_runs_on_array(gray_image_array):
    pipe = PreprocessPipeline(input_size=48)
    out = pipe(gray_image_array)
    assert out.shape == (48, 48, 3)
    assert out.dtype == np.float32


def test_pipeline_runs_on_file(png_image):
    pipe = PreprocessPipeline(input_size=32)
    out = pipe(png_image)
    assert out.shape == (32, 32, 3)


def test_pipeline_steps_traceable():
    pipe = PreprocessPipeline()
    steps = pipe.steps()
    assert steps == ["load_gray", "denoise", "clahe", "resize", "normalize", "replicate_channels"]


def test_to_tensor_requires_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        import pytest
        with pytest.raises(RuntimeError):
            to_tensor(np.zeros((4, 4, 3), dtype=np.float32))
    else:
        t = to_tensor(np.zeros((4, 4, 3), dtype=np.float32))
        assert tuple(t.shape) == (3, 4, 4)


def test_self_check_single():
    a3 = np.zeros((4, 4, 3), dtype=np.float32)
    assert self_check_single(a3) is True
    a3[..., 2] = 1.0
    assert self_check_single(a3) is False
    assert self_check_single(np.zeros((4, 4), dtype=np.float32)) is False
