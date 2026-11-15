"""Tests for predict.py."""

from __future__ import annotations

import numpy as np
import pytest

from weld_defect.predict import WeldPredictor


def test_predict_returns_contract(tiny_checkpoint, sample_image_uint8):
    """predict() returns a dict matching the API contract."""
    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    result = p.predict(sample_image_uint8, image_name="test.png")

    # API contract (docs/API.md §2.3).
    assert result["image"] == "test.png"
    assert result["predicted_class"] in ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "top_k" in result
    assert len(result["top_k"]) == 6
    assert result["inference_ms"] > 0
    assert result["model"].startswith("custom_cnn")
    assert result["decision_support_only"] is True


def test_predict_top_k_sorted(tiny_checkpoint, sample_image_uint8):
    """top_k is sorted by probability descending."""
    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    result = p.predict(sample_image_uint8)
    probs = [t["probability"] for t in result["top_k"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_top_k_sums_to_one(tiny_checkpoint, sample_image_uint8):
    """top_k probabilities sum to ~1.0."""
    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    result = p.predict(sample_image_uint8)
    total = sum(t["probability"] for t in result["top_k"])
    assert abs(total - 1.0) < 1e-4


def test_predict_from_file(tiny_checkpoint, tmp_path, sample_image_uint8):
    """predict() accepts a file path."""
    import cv2
    img_path = tmp_path / "weld.png"
    cv2.imwrite(str(img_path), sample_image_uint8)

    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    result = p.predict(str(img_path), image_name="weld.png")
    assert result["image"] == "weld.png"
    assert "predicted_class" in result


def test_predict_batch(tiny_checkpoint, sample_image_uint8):
    """predict_batch returns a list of predictions."""
    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    results = p.predict_batch([sample_image_uint8, sample_image_uint8])
    assert len(results) == 2
    assert all("predicted_class" in r for r in results)


def test_predict_no_checkpoint_degraded():
    """Predictor without a checkpoint runs in degraded mode."""
    p = WeldPredictor(checkpoint_path=None, architecture="custom_cnn", input_size=64)
    assert p.model_loaded is False
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    result = p.predict(img)
    assert "predicted_class" in result
    assert result["decision_support_only"] is True


def test_predict_reproducible(tiny_checkpoint, sample_image_uint8):
    """Same image → same prediction."""
    p = WeldPredictor(checkpoint_path=str(tiny_checkpoint), architecture="custom_cnn", input_size=64)
    r1 = p.predict(sample_image_uint8)
    r2 = p.predict(sample_image_uint8)
    assert r1["predicted_class"] == r2["predicted_class"]
    assert abs(r1["confidence"] - r2["confidence"]) < 1e-5
