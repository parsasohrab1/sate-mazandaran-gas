"""Radiograph preprocessing for weld-defect classification.

Pipeline (see configs/train.yaml → augmentation + data sections):

    raw DICOM/PNG  →  grayscale  →  denoise  →  CLAHE  →  resize  →  normalize
                    →  (replicate to 3 channels for pretrained backbones)  →  tensor

The functions are written so that the pure-numpy/Pillow steps work even when
PyTorch is absent (useful for fast unit tests and CI). Only ``to_tensor``
requires torch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def load_image_gray(path: str | Path) -> np.ndarray:
    """Load an image and convert it to a float32 grayscale array in [0, 1]."""
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def denoise(img: np.ndarray, strength: int = 5) -> np.ndarray:
    """Non-local-means denoise; falls back to median blur if cv2 is missing."""
    if _HAS_CV2:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        out = cv2.fastNlMeansDenoising(u8, None, h=strength, templateWindowSize=7, searchWindowSize=21)
        return out.astype(np.float32) / 255.0
    # numpy fallback: 3x3 median
    from scipy.ndimage import median_filter
    return median_filter(img, size=3).astype(np.float32)


def clahe(img: np.ndarray, clip_limit: float = 4.0, tile: int = 8) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization — boosts faint defects."""
    if _HAS_CV2:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(u8)
        return eq.astype(np.float32) / 255.0
    # numpy fallback: global histogram equalization
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
    return cdf[u8].astype(np.float32) / 255.0


def resize_to(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Bilinear resize to (size, size)."""
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def normalize(img: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Standardize to roughly zero mean / unit variance."""
    return (img - mean) / std


def replicate_channels(img: np.ndarray, channels: int = 3) -> np.ndarray:
    """Repeat a single-channel image into ``channels`` (for pretrained backbones)."""
    if img.ndim == 2:
        return np.stack([img] * channels, axis=-1)
    return img


def to_tensor(img: np.ndarray) -> "torch.Tensor":
    """HWC float32 ndarray → CHW tensor. Requires torch."""
    if not _HAS_TORCH:
        raise RuntimeError("to_tensor requires torch (not installed in this env).")
    if img.ndim == 2:
        img = replicate_channels(img, 3)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(img).float()


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
class PreprocessPipeline:
    """Composes the steps above. Callable on a file path or an ndarray."""

    def __init__(
        self,
        input_size: int = 224,
        denoise_strength: int = 5,
        clahe_clip: float = 4.0,
        mean: float = 0.5,
        std: float = 0.5,
        channels: int = 3,
    ) -> None:
        self.input_size = input_size
        self.denoise_strength = denoise_strength
        self.clahe_clip = clahe_clip
        self.mean = mean
        self.std = std
        self.channels = channels

    def __call__(self, source: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(source, (str, Path)):
            img = load_image_gray(source)
        else:
            img = source.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            # Collapse 3-channel (e.g. from an augmenter) back to single channel
            # for the radiograph-specific denoise/CLAHE steps.
            if img.ndim == 3:
                if img.shape[-1] == 1:
                    img = img[..., 0]
                else:
                    img = img[..., 0]  # radiographs are grayscale; take first channel
        img = denoise(img, self.denoise_strength)
        img = clahe(img, self.clahe_clip)
        img = resize_to(img, self.input_size)
        img = normalize(img, self.mean, self.std)
        img = replicate_channels(img, self.channels)
        return img

    def steps(self) -> list[str]:
        """Return the ordered step names (for logging / PAL traceability)."""
        return ["load_gray", "denoise", "clahe", "resize", "normalize", "replicate_channels"]


def preprocess_directory(
    src_dir: str | Path,
    dst_dir: str | Path,
    pipeline: PreprocessPipeline | None = None,
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> int:
    """Bulk-preprocess all images in ``src_dir`` into ``dst_dir`` as PNG.

    Returns the number of images written. Used by ``scripts/prepare_data.py``.
    """
    from weld_defect.utils import ensure_dir

    pipeline = pipeline or PreprocessPipeline()
    dst = ensure_dir(dst_dir)
    n = 0
    for p in sorted(Path(src_dir).iterdir()):
        if p.suffix.lower() not in exts:
            continue
        arr = pipeline(p)  # HWC float
        u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.min() < 0 else (arr * 255).astype(np.uint8)
        Image.fromarray(u8[0] if u8.ndim == 3 and self_check_single(u8) else u8).save(dst / f"{p.stem}.png")
        n += 1
    return n


def self_check_single(arr: np.ndarray) -> bool:
    """True if all channels of ``arr`` are identical (so we can collapse to 2D)."""
    if arr.ndim != 3:
        return False
    return bool(np.allclose(arr[..., 0], arr[..., -1]))
