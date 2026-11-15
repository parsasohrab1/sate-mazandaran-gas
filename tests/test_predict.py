"""Prediction module tests — requires torch."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from weld_defect.models.factory import save_model
from weld_defect.models import CustomCNN
from weld_defect.predict import predict

CLASS_NAMES = ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]


def test_predict_single(tmp_path, png_image):
    ckpt = tmp_path / "m.pt"
    m = CustomCNN(num_classes=6, input_size=224)
    save_model(m, ckpt, meta={"architecture": "custom_cnn", "num_classes": 6,
                              "input_size": 224, "class_names": CLASS_NAMES})
    out = predict(png_image, checkpoint=ckpt, class_names=CLASS_NAMES, top_k=3)
    assert out["predicted_class"] in CLASS_NAMES
    assert 0.0 <= out["confidence"] <= 1.0
    assert len(out["top_k"]) <= 3
    assert out["decision_support_only"] is True
    assert out["inference_ms"] >= 0.0


def test_predict_missing_checkpoint(tmp_path, png_image):
    with pytest.raises(FileNotFoundError):
        predict(png_image, checkpoint=tmp_path / "nope.pt")
