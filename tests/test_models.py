"""Tests for models."""

from __future__ import annotations

import pytest
import torch

from weld_defect.models.custom_cnn import CustomCNN
from weld_defect.models.ensemble import EnsembleModel
from weld_defect.models.factory import build_model, load_model, save_model


def test_custom_cnn_forward():
    """CustomCNN forward pass produces correct output shape."""
    model = CustomCNN(num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 6)


def test_custom_cnn_param_count():
    """CustomCNN has a reasonable lightweight parameter count."""
    model = CustomCNN(num_classes=6)
    n = sum(p.numel() for p in model.parameters())
    # Lightweight CNN — should be under 1M params for fast CPU inference.
    assert 100_000 < n < 1_000_000, f"Param count {n} outside expected range for a lightweight CNN"


def test_build_model_custom_cnn():
    """build_model('custom_cnn') returns CustomCNN."""
    model = build_model("custom_cnn", num_classes=6, pretrained=False)
    assert isinstance(model, CustomCNN)


def test_build_model_resnet50():
    """build_model('resnet50') returns a ResNet50."""
    from weld_defect.models.resnet import ResNet50Classifier
    model = build_model("resnet50", num_classes=6, pretrained=False)
    assert isinstance(model, ResNet50Classifier)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 6)


def test_build_model_efficientnet():
    """build_model('efficientnet_b0') returns an EfficientNet."""
    from weld_defect.models.efficientnet import EfficientNetB0Classifier
    model = build_model("efficientnet_b0", num_classes=6, pretrained=False)
    assert isinstance(model, EfficientNetB0Classifier)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 6)


def test_build_model_ensemble():
    """build_model('ensemble') returns an EnsembleModel with 3 members."""
    model = build_model(
        "ensemble",
        num_classes=6,
        pretrained=False,
        members=["resnet50", "efficientnet_b0", "custom_cnn"],
    )
    assert isinstance(model, EnsembleModel)
    assert len(model.members) == 3


def test_build_model_unknown_raises():
    """Unknown architecture raises ValueError."""
    with pytest.raises(ValueError):
        build_model("unknown_arch", num_classes=6)


def test_save_load_roundtrip(tmp_path):
    """save_model + load_model roundtrip preserves state dict."""
    model = build_model("custom_cnn", num_classes=6, pretrained=False)
    model.eval()  # set to eval mode before saving
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out_before = model(x)

    ckpt = tmp_path / "model.pt"
    save_model(model, ckpt)
    assert ckpt.exists()

    loaded = load_model(ckpt, architecture="custom_cnn", num_classes=6)
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)

    # Same weights → same output.
    assert torch.allclose(out_before, out_after, atol=1e-4), \
        f"Outputs differ: {out_before.flatten()[:5]} vs {out_after.flatten()[:5]}"


def test_ensemble_forward():
    """Ensemble forward produces correct shape."""
    model = build_model(
        "ensemble",
        num_classes=6,
        pretrained=False,
        members=["custom_cnn"],  # single member for speed
    )
    x = torch.randn(1, 3, 64, 64)
    # CustomCNN uses AdaptiveAvgPool so any size works.
    out = model(x)
    assert out.shape == (1, 6)


def test_ensemble_strategy_invalid():
    """Invalid ensemble strategy raises ValueError."""
    from weld_defect.models.custom_cnn import CustomCNN
    with pytest.raises(ValueError):
        EnsembleModel([CustomCNN(num_classes=6)], strategy="invalid")


def test_load_model_missing_file_raises(tmp_path):
    """Missing checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pt", architecture="custom_cnn")
