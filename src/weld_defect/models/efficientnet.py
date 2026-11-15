"""EfficientNet backbone (b0-b3) with a project head.

Uses ``timm`` if available (broader variant coverage and more recent
pretrained weights); falls back to torchvision's efficientnet_b0 otherwise.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False


class EfficientNetBackbone(nn.Module):
    def __init__(self, num_classes: int = 6, input_size: int = 224, pretrained: bool = True,
                 variant: str = "efficientnet_b0") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.variant = variant

        if _HAS_TIMM:
            self.backbone = timm.create_model(variant, pretrained=pretrained, num_classes=0, global_pool="avg")
            in_feats = self.backbone.num_features
        else:
            from torchvision import models
            if variant != "efficientnet_b0":
                raise RuntimeError(
                    f"timm not installed; only efficientnet_b0 is supported, got {variant}"
                )
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_feats = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
