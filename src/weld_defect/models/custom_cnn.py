"""CustomCNN — lightweight 6-conv + 2-fc baseline (~3.2M params).

Trained from scratch on radiographs; used as the fast path and as a diversity
member of the ensemble (configs/model_catalog.yaml).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CustomCNN(nn.Module):
    """6-conv + 2-fc CNN for weld defect classification."""

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        input_size: int = 224,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # 6 conv blocks with BatchNorm + ReLU + occasional pooling.
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 → 112

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 → 56

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 → 28
        )

        # Adaptive pool to handle any input size.
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
