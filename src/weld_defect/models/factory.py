"""Model factory: construction, serialization, and loading.

Keeps the rest of the codebase decoupled from concrete model classes. The
factory records the construction kwargs alongside the state dict so a checkpoint
can be reconstructed without the original config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from weld_defect.models.custom_cnn import CustomCNN
from weld_defect.models.resnet import ResNetBackbone
from weld_defect.models.efficientnet import EfficientNetBackbone
from weld_defect.models.ensemble import EnsembleModel


def build_model(
    architecture: str,
    num_classes: int = 6,
    input_size: int = 224,
    pretrained: bool = True,
    **kwargs: Any,
) -> torch.nn.Module:
    """Construct a model by architecture name."""
    if architecture == "custom_cnn":
        return CustomCNN(num_classes, input_size, pretrained=False)
    if architecture in ("resnet18", "resnet34", "resnet50"):
        return ResNetBackbone(num_classes, input_size, pretrained, variant=architecture)
    if architecture in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"):
        return EfficientNetBackbone(num_classes, input_size, pretrained, variant=architecture)
    if architecture == "ensemble":
        return EnsembleModel(
            members=kwargs.get("members", ["resnet50", "efficientnet_b0", "custom_cnn"]),
            num_classes=num_classes,
            input_size=input_size,
            pretrained=pretrained,
            strategy=kwargs.get("ensemble_strategy", "softmax_avg"),
            weights=kwargs.get("weights"),
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def build_from_config(cfg) -> torch.nn.Module:
    """Build a model from a ``ModelConfig`` (see weld_defect.config)."""
    return build_model(
        architecture=cfg.architecture,
        num_classes=getattr(cfg, "num_classes", 6),
        input_size=getattr(cfg, "input_size", 224),
        pretrained=cfg.pretrained,
        members=list(getattr(cfg, "members", [])),
        ensemble_strategy=cfg.ensemble_strategy,
    )


def save_model(model: torch.nn.Module, path: str | Path, meta: dict | None = None) -> Path:
    """Save a model with construction metadata in a single file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "meta": meta or {},
        "architecture": getattr(model, "variant", model.__class__.__name__),
    }
    if isinstance(model, EnsembleModel):
        payload["meta"].update({
            "kind": "ensemble",
            "members": model.member_names,
            "strategy": model.strategy,
            "weights": model.ensemble_weights.tolist(),
        })
    torch.save(payload, p)
    return p


def load_model(path: str | Path, map_location: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Load a model saved by :func:`save_model`. Returns (model, meta)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {p}")
    payload = torch.load(p, map_location=map_location, weights_only=False)
    meta = payload.get("meta", {})
    if meta.get("kind") == "ensemble":
        model = EnsembleModel(
            members=meta["members"],
            strategy=meta.get("strategy", "softmax_avg"),
            weights=meta.get("weights"),
            num_classes=meta.get("num_classes", 6),
            input_size=meta.get("input_size", 224),
            pretrained=False,
        )
    else:
        arch = meta.get("architecture", payload.get("architecture", "custom_cnn"))
        model = build_model(arch, num_classes=meta.get("num_classes", 6),
                            input_size=meta.get("input_size", 224), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta
