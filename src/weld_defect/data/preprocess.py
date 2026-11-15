"""Image preprocessing pipeline: denoise → CLAHE → resize → normalize.

Implements SRS FR-1.2.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PreprocessPipeline:
    """Reproducible preprocessing pipeline for radiographic images.

    Steps (in order):
      1. Convert to grayscale if needed.
      2. Denoise (cv2.fastNlMeansDenoising).
      3. CLAHE (contrast-limited adaptive histogram equalization).
      4. Resize to ``input_size`` × ``input_size``.
      5. Normalize to [0, 1].
      6. Replicate to 3 channels (for pretrained backbones).
    """

    def __init__(
        self,
        input_size: int = 224,
        denoise: bool = True,
        clahe: bool = True,
        clahe_clip_limit: float = 4.0,
        clahe_grid_size: int = 8,
    ) -> None:
        self.input_size = input_size
        self.denoise = denoise
        self.clahe = clahe
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_grid_size, clahe_grid_size))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Process a single image; return HWC float32 array in [0,1], 3 channels."""
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to PreprocessPipeline")

        # Convert to grayscale.
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]
            else:
                raise ValueError(f"Unsupported channel count: {image.shape[2]}")
        elif image.ndim != 2:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # uint8 for denoise/CLAHE.
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if self.denoise:
            image = cv2.fastNlMeansDenoising(image, None, h=7, templateWindowSize=7, searchWindowSize=21)

        if self.clahe:
            image = self._clahe.apply(image)

        # Resize.
        if image.shape[0] != self.input_size or image.shape[1] != self.input_size:
            image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        # Normalize to [0,1].
        image = image.astype(np.float32) / 255.0

        # Replicate to 3 channels.
        image = np.stack([image, image, image], axis=-1)  # HWC
        return image


def preprocess_image(path: str | Path, input_size: int = 224) -> np.ndarray:
    """Load an image from disk and run the preprocessing pipeline.

    Supports PNG, JPG, BMP, TIFF (via OpenCV).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    # cv2.imread doesn't support unicode paths on Windows; use cv2.imdecode.
    raw = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {p}")

    pipeline = PreprocessPipeline(input_size=input_size)
    return pipeline(image)
