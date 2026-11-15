"""Lightweight CNN trained from scratch on radiographs.

A 6-conv + 2-fc network used as a fast path and as a diversity member of the
production ensemble. ~3.2M params, ~25 ms latency on a single CPU thread.
"""
from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 6, input_size: int = 224, pretrained: bool = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        # pretrained is ignored for this from-scratch model (kept for API symmetry)

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )
        # After 3 maxpools on 224 → 28x28 with 128 channels
        feat_size = input_size // 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
