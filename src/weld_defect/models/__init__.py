"""Model architectures for weld-defect classification.

All models share the same constructor signature
``Model(num_classes, input_size, pretrained)`` and the same forward
``(B, 3, H, W) -> (B, num_classes)`` logits, so they can be freely combined in
an ensemble (see ``EnsembleModel``).
"""
from weld_defect.models.custom_cnn import CustomCNN
from weld_defect.models.resnet import ResNetBackbone
from weld_defect.models.efficientnet import EfficientNetBackbone
from weld_defect.models.ensemble import EnsembleModel
from weld_defect.models.factory import build_model, build_from_config, save_model, load_model

__all__ = [
    "CustomCNN",
    "ResNetBackbone",
    "EfficientNetBackbone",
    "EnsembleModel",
    "build_model",
    "build_from_config",
    "save_model",
    "load_model",
]
