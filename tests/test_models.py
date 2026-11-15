"""Model architecture tests — requires torch (skipped if absent)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from weld_defect.models import (
    CustomCNN, ResNetBackbone, EfficientNetBackbone, EnsembleModel,
    build_model, build_from_config, save_model, load_model,
)
from weld_defect.config import ModelConfig


def _dummy_batch(size=32):
    return torch.randn(2, 3, size, size)


def test_custom_cnn_forward():
    m = CustomCNN(num_classes=6, input_size=32)
    out = m(_dummy_batch(32))
    assert out.shape == (2, 6)
    assert torch.isfinite(out).all()


def test_resnet_forward():
    m = ResNetBackbone(num_classes=6, input_size=32, pretrained=False, variant="resnet18")
    out = m(_dummy_batch(32))
    assert out.shape == (2, 6)


def test_efficientnet_forward():
    try:
        m = EfficientNetBackbone(num_classes=6, input_size=32, pretrained=False, variant="efficientnet_b0")
    except Exception as exc:
        pytest.skip(f"efficientnet unavailable without timm/torchvision: {exc}")
    out = m(_dummy_batch(32))
    assert out.shape == (2, 6)


def test_ensemble_forward_returns_probs():
    m = EnsembleModel(members=["custom_cnn"], num_classes=6, input_size=32, pretrained=False,
                      strategy="softmax_avg")
    out = m(_dummy_batch(32))
    # EnsembleModel.forward returns probabilities
    assert out.shape == (2, 6)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4)


def test_build_model_unknown_raises():
    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_build_from_config():
    cfg = ModelConfig(architecture="custom_cnn", members=["custom_cnn"], pretrained=False,
                      ensemble_strategy="softmax_avg")
    m = build_from_config(cfg)
    assert isinstance(m, CustomCNN)


def test_save_and_load_roundtrip(tmp_path):
    m = CustomCNN(num_classes=6, input_size=32)
    ckpt = tmp_path / "model.pt"
    save_model(m, ckpt, meta={"architecture": "custom_cnn", "num_classes": 6, "input_size": 32})
    m2, meta = load_model(ckpt)
    assert meta.get("architecture") == "custom_cnn"
    # Same output for same input (after eval)
    m.eval(); m2.eval()
    x = _dummy_batch(32)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-5)
