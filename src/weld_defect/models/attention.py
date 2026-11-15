"""SE (Squeeze-and-Excitation) attention blocks.

SE blocks recalibrate channel-wise feature responses by learning a per-channel
attention weight. Adding SE blocks to CustomCNN improves focus on diagnostically
important regions of the radiograph with negligible parameter cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Args:
        channels: Number of input/output channels.
        reduction: Reduction ratio for the excitation MLP (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.squeeze(x).view(b, c)
        s = self.excite(s).view(b, c, 1, 1)
        return x * s.expand_as(x)


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion via parallel branches at different resolutions.

    Processes the input at 3 scales (full, half, quarter) and fuses the
    features. Helps detect defects of varying sizes (thin cracks vs large slag).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch_full = nn.Conv2d(in_channels, out_channels // 3, 3, padding=1)
        self.branch_half = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.branch_quarter = nn.Sequential(
            nn.AvgPool2d(4),
            nn.Conv2d(in_channels, out_channels - 2 * (out_channels // 3), 3, padding=1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch_full(x), self.branch_half(x), self.branch_quarter(x)], dim=1)
