"""Single-image inference.

Implements SRS FR-4 (inference API) at the library level — the FastAPI app
in ``weld_defect.api.app`` wraps this.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import CLASS_NAMES
from .data.preprocess import PreprocessPipeline, preprocess_image
from .models.factory import build_model, load_model

SERVICE_VERSION = "1.0.0"


class WeldPredictor:
    """Load a trained model and run inference on individual images."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        architecture: str = "ensemble",
        input_size: int = 224,
        device: str = "cpu",
        num_classes: int = 6,
        class_names: list[str] | None = None,
    ) -> None:
        self.input_size = input_size
        self.device = torch.device(device)
        self.class_names = class_names or list(CLASS_NAMES)
        self.num_classes = num_classes

        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                self.model = load_model(
                    checkpoint_path, architecture=architecture,
                    num_classes=num_classes, map_location=self.device,
                ).to(self.device)
                self.model_loaded = True
                self.architecture_name = architecture
            except Exception:
                self.model = build_model(architecture, num_classes=num_classes, pretrained=False).to(self.device)
                self.model_loaded = False
                self.architecture_name = architecture
        else:
            # No checkpoint — build untrained model (degraded mode).
            self.model = build_model(architecture, num_classes=num_classes, pretrained=False).to(self.device)
            self.model_loaded = False
            self.architecture_name = architecture

        self.model.eval()
        self._pipeline = PreprocessPipeline(input_size=input_size)

    def _to_tensor(self, image: np.ndarray | str) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = preprocess_image(image, self.input_size)
        else:
            image = self._pipeline(image)

        # HWC float32 [0,1] → CHW normalized.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW
        return torch.from_numpy(image.astype(np.float32)).unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray | str, image_name: str = "image") -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: Either a file path or an HWC uint8/float numpy array.
            image_name: Name to use in the response (for logging).

        Returns:
            Dict matching the API contract in docs/API.md §2.3.
        """
        t0 = time.perf_counter()
        tensor = self._to_tensor(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        inference_ms = (time.perf_counter() - t0) * 1000

        best_idx = int(probs.argmax())
        top_k = [
            {"class_name": self.class_names[i], "probability": float(probs[i])}
            for i in np.argsort(probs)[::-1]
        ]

        return {
            "image": image_name,
            "predicted_class": self.class_names[best_idx],
            "confidence": float(probs[best_idx]),
            "top_k": top_k,
            "inference_ms": float(inference_ms),
            "model": f"{self.architecture_name}-v{SERVICE_VERSION}",
            "decision_support_only": True,
        }

    def predict_batch(self, images: list[np.ndarray | str], image_names: list[str] | None = None) -> list[dict[str, Any]]:
        """Run inference on a batch of images."""
        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(images))]
        return [self.predict(img, name) for img, name in zip(images, image_names)]
