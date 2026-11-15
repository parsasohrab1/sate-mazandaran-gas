"""Ensemble of heterogeneous backbones.

Combines a CustomCNN (texture + geometry, lightweight), a ResNet50 (texture
bias), and an EfficientNet-B0 (compound scaling) to decorrelate errors.

Strategies:
    softmax_avg — average the softmax probabilities of all members (default)
    voting      — majority vote of argmax predictions
    stacking    — weighted softmax; weights learned on the val set
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from weld_defect.models.custom_cnn import CustomCNN
from weld_defect.models.resnet import ResNetBackbone
from weld_defect.models.efficientnet import EfficientNetBackbone


_MEMBER_CTORS = {
    "custom_cnn": lambda **kw: CustomCNN(num_classes=kw["num_classes"], input_size=kw["input_size"], pretrained=False),
    "resnet50": lambda **kw: ResNetBackbone(num_classes=kw["num_classes"], input_size=kw["input_size"], pretrained=kw["pretrained"], variant="resnet50"),
    "resnet18": lambda **kw: ResNetBackbone(num_classes=kw["num_classes"], input_size=kw["input_size"], pretrained=kw["pretrained"], variant="resnet18"),
    "efficientnet_b0": lambda **kw: EfficientNetBackbone(num_classes=kw["num_classes"], input_size=kw["input_size"], pretrained=kw["pretrained"], variant="efficientnet_b0"),
}


class EnsembleModel(nn.Module):
    def __init__(
        self,
        members: list[str],
        num_classes: int = 6,
        input_size: int = 224,
        pretrained: bool = True,
        strategy: str = "softmax_avg",
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strategy = strategy
        self.member_names = list(members)
        self.members = nn.ModuleList([
            _MEMBER_CTORS[name](num_classes=num_classes, input_size=input_size, pretrained=pretrained)
            for name in members
        ])
        n = len(self.members)
        if weights is None:
            weights = [1.0 / n] * n
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("ensemble_weights", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strategy == "voting":
            preds = [m(x).argmax(dim=1) for m in self.members]
            one_hot = torch.stack([F.one_hot(p, self.num_classes).float() for p in preds], dim=0)
            return one_hot.mean(dim=0)  # pseudo-probabilities

        # softmax_avg / stacking both average softmax probs (weights differ)
        probs = 0.0
        for w, m in zip(self.ensemble_weights, self.members):
            probs = probs + w * F.softmax(m(x), dim=1)
        return probs  # probabilities, not logits — callers must NOT apply softmax again

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged logits (use for training individual members instead)."""
        return sum(m(x) for m in self.members) / len(self.members)

    def member_forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        """Forward a single member (used during per-member pretraining)."""
        return self.members[idx](x)
