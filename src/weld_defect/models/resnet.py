"""ResNet50 classifier — transfer-learning baseline for texture defects.

~25.6M params, good at porosity and slag (configs/model_catalog.yaml).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResNet50Classifier(nn.Module):
    """ResNet50 with a custom classification head."""

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError:
            from torchvision.models import resnet50
            ResNet50_Weights = None

        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = resnet50(weights=weights)
        else:  # pragma: no cover
            self.backbone = resnet50(pretrained=pretrained)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
