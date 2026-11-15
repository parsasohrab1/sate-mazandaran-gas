"""Model factory — build, save, and load models.

Implements SRS FR-2.1 (support for CustomCNN, ResNet50, EfficientNet-B0, EfficientNetV2-S, Ensemble).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .custom_cnn import CustomCNN
from .custom_cnn_v2 import CustomCNNv2
from .efficientnet import EfficientNetB0Classifier
from .efficientnet_v2 import EfficientNetV2SClassifier
from .ensemble import EnsembleModel
from .resnet import ResNet50Classifier

_REGISTRY: dict[str, type[nn.Module]] = {
    "custom_cnn": CustomCNN,
    "customcnn": CustomCNN,
    "cnn": CustomCNN,
    "custom_cnn_v2": CustomCNNv2,
    "customcnn_v2": CustomCNNv2,
    "cnn_v2": CustomCNNv2,
    "resnet50": ResNet50Classifier,
    "resnet": ResNet50Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "efficientnet": EfficientNetB0Classifier,
    "efficientnetv2_s": EfficientNetV2SClassifier,
    "efficientnet_v2_s": EfficientNetV2SClassifier,
    "efficientnet_v2": EfficientNetV2SClassifier,
    "ensemble": EnsembleModel,
}


def build_model(
    architecture: str,
    num_classes: int = 6,
    pretrained: bool = True,
    config: dict[str, Any] | None = None,
    members: list[str] | None = None,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a model by architecture name.

    Args:
        architecture: One of 'custom_cnn', 'resnet50', 'efficientnet_b0', 'ensemble'.
        num_classes: Number of output classes (default 6 per API 1104).
        pretrained: Use ImageNet pretrained weights for transfer-learning backbones.
        config: Optional dict with 'members', 'ensemble_strategy', etc.
        members: For ensemble — list of member architecture names.
        dropout: Dropout rate for the classifier head.

    Returns:
        Instantiated nn.Module.
    """
    arch_key = architecture.lower().strip()
    config = config or {}

    if arch_key == "ensemble":
        member_names = members or config.get("members", ["resnet50", "efficientnet_b0", "custom_cnn"])
        strategy = config.get("ensemble_strategy", "softmax_avg")
        member_models = [
            build_model(name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
            for name in member_names
        ]
        return EnsembleModel(member_models, strategy=strategy)

    cls = _REGISTRY.get(arch_key)
    if cls is None:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Known: {sorted(set(_REGISTRY.keys()))}"
        )

    # Pretrained flag only applies to transfer-learning backbones.
    if cls in (CustomCNN, CustomCNNv2):
        return cls(num_classes=num_classes, dropout=dropout)
    return cls(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def save_model(model: nn.Module, path: str | Path, formats: list[str] | None = None) -> list[Path]:
    """Save a model to ``path`` (and optionally ONNX).

    Args:
        model: Model to save.
        path: Output .pt path.
        formats: List of formats to export ('pt', 'onnx').

    Returns:
        List of paths written.
    """
    formats = formats or ["pt"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "pt" in formats:
        # Save state_dict + metadata.
        torch.save({
            "state_dict": model.state_dict(),
            "architecture": getattr(model, "__class__").__name__,
        }, p)
        written.append(p)

    if "onnx" in formats:
        try:
            onnx_path = p.with_suffix(".onnx")
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, dummy, str(onnx_path), opset_version=14)
            written.append(onnx_path)
        except Exception:
            pass  # ONNX export is best-effort.

    return written


def load_model(
    path: str | Path,
    architecture: str = "ensemble",
    num_classes: int = 6,
    map_location: str | torch.device = "cpu",
    members: list[str] | None = None,
) -> nn.Module:
    """Load a model from a checkpoint.

    Args:
        path: Path to the .pt checkpoint.
        architecture: Architecture name (used to rebuild the model shell).
        num_classes: Number of classes.
        map_location: Device to map tensors to.
        members: For ensemble — member architecture names.

    Returns:
        Loaded model in eval mode.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    model = build_model(architecture, num_classes=num_classes, pretrained=False, members=members)

    ckpt = torch.load(p, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    else:
        state = ckpt

    # Try strict load; fall back to non-strict (some keys may not match if
    # the checkpoint was saved with a slightly different head).
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)

    model.eval()
    return model
