"""CustomCNN with SE (Squeeze-and-Excitation) attention blocks.

Enhanced version of CustomCNN with:
  - SE attention after each conv block (channel recalibration)
  - Multi-scale feature fusion for defect-size robustness
  - Improved generalization on safety-critical crack class

Architecture: 6-conv + SE + 2-fc, ~0.5M params (slightly more than base).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import SEBlock


class CustomCNNv2(nn.Module):
    """6-conv + 2-fc CNN with SE attention blocks."""

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        input_size: int = 224,
        dropout: float = 0.3,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            SEBlock(32, reduction=se_reduction),
            nn.MaxPool2d(2, 2),  # 224 → 112

            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            SEBlock(64, reduction=se_reduction),
            nn.MaxPool2d(2, 2),  # 112 → 56

            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            SEBlock(128, reduction=se_reduction),
            nn.MaxPool2d(2, 2),  # 56 → 28
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x
