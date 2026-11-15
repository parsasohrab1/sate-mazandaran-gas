"""Augmentation policies for radiographs.

Wraps Albumentations when available; otherwise falls back to a small numpy
policy so the package is importable in minimal CI environments. The policy is
declared in ``configs/train.yaml`` under ``augmentation.ops``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import albumentations as A
    _HAS_ALB = True
except ImportError:  # pragma: no cover
    _HAS_ALB = False


class Augmenter:
    """Callable augmenter.

    Parameters
    ----------
    ops : list of dicts
        Each dict has a ``name`` (Albumentations transform name) and a ``p``
        (probability) plus transform-specific kwargs.
    input_size : int
        Final image size, used to compose a final Resize.
    """

    def __init__(self, ops: list[dict[str, Any]] | None = None, input_size: int = 224) -> None:
        self.input_size = input_size
        self.ops = ops or []
        self._fn = self._build_albumentations() if _HAS_ALB else self._build_numpy()

    # ----- Albumentations backend -----
    def _build_albumentations(self) -> "A.Compose":
        transforms = []
        for spec in self.ops:
            name = spec.get("name")
            p = spec.get("p", 0.5)
            kwargs = {k: v for k, v in spec.items() if k not in ("name", "p")}
            cls = getattr(A, name, None)
            if cls is None:
                continue
            transforms.append(cls(p=p, **kwargs))
        transforms.append(A.Resize(self.input_size, self.input_size))
        return A.Compose(transforms)

    # ----- Numpy fallback (deterministic, p ignored) -----
    def _build_numpy(self):
        rng = np.random.default_rng(42)

        def fn(image: np.ndarray) -> np.ndarray:
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[..., 0]
            # random horizontal flip
            if rng.random() < 0.5:
                image = image[:, ::-1]
            # random brightness/contrast jitter
            b = 1.0 + rng.uniform(-0.2, 0.2)
            c = 1.0 + rng.uniform(-0.2, 0.2)
            image = np.clip((image - 0.5) * c + 0.5, 0, 1)
            image = np.clip(image * b, 0, 1)
            # resize via PIL
            from PIL import Image as PILImage
            pil = PILImage.fromarray((image * 255).astype(np.uint8))
            pil = pil.resize((self.input_size, self.input_size), PILImage.BILINEAR)
            return np.asarray(pil, dtype=np.float32) / 255.0

        return fn

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = self._fn(image=image) if _HAS_ALB else self._fn(image)
        if _HAS_ALB:
            out = out["image"]
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=-1)
        elif out.ndim == 3 and out.shape[-1] == 1:
            out = np.concatenate([out, out, out], axis=-1)
        return np.ascontiguousarray(out)


def build_augmenter(config) -> Augmenter:
    """Build an Augmenter from an ``AugmentConfig`` (see weld_defect.config)."""
    if not getattr(config, "enabled", True):
        return Augmenter(ops=[], input_size=224)
    return Augmenter(ops=list(config.ops), input_size=224)
