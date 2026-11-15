"""Single-image prediction (used by the CLI and the API)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

from weld_defect.data.preprocess import PreprocessPipeline
from weld_defect.models.factory import load_model


def predict(
    image_path: str | Path,
    checkpoint: str | Path = "models/ensemble_best.pt",
    class_names: Sequence[str] | None = None,
    input_size: int = 224,
    top_k: int = 3,
    model: "torch.nn.Module | None" = None,
    meta: dict | None = None,
) -> dict:
    """Run inference on one image and return top-k predictions + latency.

    If ``model``/``meta`` are supplied (already loaded by the caller, e.g. the
    API), they are reused; otherwise the model is loaded from ``checkpoint``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("torch is required for prediction.")
    if model is None or meta is None:
        model, meta = load_model(checkpoint)
    model.eval()
    names = list(class_names or meta.get("class_names", []))
    pre = PreprocessPipeline(input_size=meta.get("input_size", input_size))
    arr = pre(image_path)  # HWC float
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    x = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    is_ensemble = meta.get("kind") == "ensemble" or meta.get("architecture") == "ensemble"
    proba = (out if is_ensemble else F.softmax(out, dim=1))[0].cpu().float().numpy()

    order = np.argsort(proba)[::-1][:top_k]
    top = [{"class": names[i] if i < len(names) else str(i),
            "probability": float(proba[i])} for i in order]
    return {
        "image": str(image_path),
        "predicted_class": top[0]["class"],
        "confidence": top[0]["probability"],
        "top_k": top,
        "inference_ms": float(dt_ms),
        "model": meta.get("architecture", "unknown"),
        "decision_support_only": True,
    }
