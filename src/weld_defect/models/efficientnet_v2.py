"""EfficientNetV2-S classifier — replaces EfficientNet-B0 for better accuracy.

~20M params (4× more than B0) but:
  - Fused-MBConv blocks → faster training convergence
  - Progressive learning → better generalization
  - +2-3 F1 points over B0 on transfer tasks

Uses ``timm`` for the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EfficientNetV2SClassifier(nn.Module):
    """EfficientNetV2-S with a custom classification head."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover
            raise ImportError("timm is required for EfficientNetV2-S") from e

        self.backbone = timm.create_model(
            "efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0,  # remove default head
            drop_rate=dropout,
        )
        in_features = self.backbone.num_features  # typically 1280
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
