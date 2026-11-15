"""ResNet backbone (resnet18/34/50) with a project head for weld-defect classification."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, num_classes: int = 6, input_size: int = 224, pretrained: bool = True,
                 variant: str = "resnet50") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.variant = variant
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            ctor = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50}[variant]
            if variant == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            elif variant == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        except KeyError as exc:
            raise ValueError(f"Unsupported resnet variant: {variant}") from exc

        self.backbone = ctor(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
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
