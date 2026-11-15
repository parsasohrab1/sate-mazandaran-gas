"""Data augmentation wrapper around Albumentations.

Builds the augmentation pipeline from the YAML config in configs/train.yaml.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:  # pragma: no cover
    _HAS_ALB = False


_OP_MAP = {
    "HorizontalFlip": "HorizontalFlip",
    "VerticalFlip": "VerticalFlip",
    "Rotate": "Rotate",
    "RandomBrightnessContrast": "RandomBrightnessContrast",
    "GaussianBlur": "GaussianBlur",
    "GaussNoise": "GaussNoise",
    "CLAHE": "Clahe",
    "CoarseDropout": "CoarseDropout",
    "RandomCrop": "RandomCrop",
    "Resize": "Resize",
}


def _build_op(op_config: dict[str, Any]) -> Any:
    """Build a single Albumentations op from a config dict {name, p, ...params}."""
    if not _HAS_ALB:
        raise ImportError("albumentations is required for augmentation")

    name = op_config["name"]
    params = {k: v for k, v in op_config.items() if k not in ("name", "p")}
    p = op_config.get("p", 0.5)

    alb_name = _OP_MAP.get(name, name)
    cls = getattr(A, alb_name, None)
    if cls is None:
        # Try the original name as-is.
        cls = getattr(A, name, None)
    if cls is None:
        raise ValueError(f"Unknown Albumentations op: {name}")

    return cls(p=p, **params)


def build_augmentation(ops_config: list[dict[str, Any]], input_size: int = 224) -> Any:
    """Build an Albumentations Compose pipeline from a list of op configs."""
    if not _HAS_ALB:
        raise ImportError("albumentations is required for augmentation")

    transforms = [_build_op(op) for op in ops_config]
    # Always resize + normalize at the end.
    transforms.append(A.Resize(input_size, input_size))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return A.Compose(transforms)


def get_train_aug(ops_config: list[dict[str, Any]], input_size: int = 224) -> Any:
    """Training augmentation pipeline."""
    if not ops_config:
        return get_val_aug(input_size)
    return build_augmentation(ops_config, input_size)


def get_val_aug(input_size: int = 224) -> Any:
    """Validation/test pipeline (resize + normalize only)."""
    if not _HAS_ALB:
        raise ImportError("albumentations is required for augmentation")
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def apply_augmentation(image: np.ndarray, transform: Any) -> np.ndarray:
    """Apply an Albumentations transform to an HWC uint8/float image; return CHW tensor-ready array."""
    result = transform(image=image)
    return result["image"]
