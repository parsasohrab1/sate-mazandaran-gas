"""EfficientNet-B0 classifier — best accuracy/params trade-off.

~5.3M params, good at geometric defects (crack, lack of fusion).
Uses ``timm`` for the backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 with a custom head."""

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
            raise ImportError("timm is required for EfficientNet-B0") from e

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,  # remove default head
            drop_rate=dropout,
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
