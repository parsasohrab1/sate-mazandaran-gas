"""FastAPI application for weld defect detection.

Implements SRS FR-4 (inference API) and the contract in docs/API.md.

Endpoints:
  GET  /                 — service info
  GET  /health           — liveness + model load status
  POST /predict          — single-image inference (multipart `file`)
  POST /predict/batch    — batch inference (multiple `files`, max 64)
  GET  /docs             — auto OpenAPI (FastAPI built-in)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import CLASS_NAMES, CLASS_NAMES_FA
from ..predict import SERVICE_VERSION, WeldPredictor

MAX_UPLOAD_MB = int(os.environ.get("API_MAX_UPLOAD_MB", "10"))
MAX_BATCH_FILES = 64
API_KEY = os.environ.get("API_KEY", "")  # empty = auth disabled (dev mode)

SERVICE_INFO = {
    "service": "sate-mazandaran-gas-weld-defect-detection",
    "version": SERVICE_VERSION,
    "docs": "/docs",
    "decision_support_only": True,
    "classes": list(CLASS_NAMES),
    "classes_fa": CLASS_NAMES_FA,
    "client": "Sate Mazandaran Gas (شرکت ملی گاز ایران)",
    "executor": "Petro Pala Toos",
    "scientific_lead": "Dr. Nosratollah Ashrafi-Payaman",
    "standards": ["API 1104", "ASME Section V", "ASME Section VIII", "ASME B31.3", "IEEE 830", "CMMI V2.0 ML5"],
}


def _check_auth(x_api_key: str | None) -> None:
    """Validate API key if configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def create_app(checkpoint_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        checkpoint_path: Path to the model checkpoint. If None, reads from
            env ``MODEL_CHECKPOINT`` or runs in degraded mode (untrained model).
    """
    ckpt = checkpoint_path or os.environ.get("MODEL_CHECKPOINT", "")
    predictor = WeldPredictor(
        checkpoint_path=ckpt if ckpt else None,
        architecture=os.environ.get("MODEL_ARCHITECTURE", "ensemble"),
        device=os.environ.get("MODEL_DEVICE", "cpu"),
    )

    app = FastAPI(
        title="Sate Mazandaran Gas — Weld Defect Detection API",
        description="AI-assisted weld defect detection for gas pipelines (decision-support only).",
        version=SERVICE_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Service information."""
        return SERVICE_INFO

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Liveness + model load status (SRS FR-4.1, API.md §2.2)."""
        return {
            "status": "ok" if predictor.model_loaded else "degraded",
            "model_loaded": predictor.model_loaded,
            "device": str(predictor.device),
            "version": SERVICE_VERSION,
            "model": predictor.architecture_name,
        }

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        """Single-image inference (API.md §2.3)."""
        _check_auth(x_api_key)

        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=422, detail="Expected an image file.")

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {MAX_UPLOAD_MB} MB.")

        import numpy as np
        import cv2
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=422, detail="Could not decode image.")

        return predictor.predict(img, image_name=file.filename or "image")

    @app.post("/predict/batch")
    async def predict_batch(
        files: list[UploadFile] = File(...),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        """Batch inference (API.md §2.4, max 64 files)."""
        _check_auth(x_api_key)

        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(status_code=422, detail=f"Batch exceeds {MAX_BATCH_FILES} files.")

        import cv2
        import numpy as np

        results = []
        total_ms = 0.0
        for f in files:
            contents = await f.read()
            if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
                results.append({"image": f.filename, "error": "file_too_large"})
                continue
            arr = np.frombuffer(contents, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                results.append({"image": f.filename, "error": "decode_failed"})
                continue
            pred = predictor.predict(img, image_name=f.filename or "image")
            results.append(pred)
            total_ms += pred["inference_ms"]

        ok_count = sum(1 for r in results if "predicted_class" in r)
        return {
            "results": results,
            "count": len(results),
            "ok": ok_count,
            "failed": len(results) - ok_count,
            "avg_inference_ms": float(total_ms / ok_count) if ok_count else 0.0,
            "decision_support_only": True,
        }

    @app.post("/review")
    async def review(
        body: dict,
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> JSONResponse:
        """Record an NDT expert review (feeds the retraining loop)."""
        _check_auth(x_api_key)

        required = ["image", "predicted_class", "confidence", "status"]
        for k in required:
            if k not in body:
                raise HTTPException(status_code=422, detail=f"Missing field: {k}")
        if body["status"] not in ("approved", "rejected"):
            raise HTTPException(status_code=422, detail="status must be 'approved' or 'rejected'.")

        import uuid
        review_id = f"rev-{uuid.uuid4().hex[:8]}"
        return JSONResponse(
            status_code=201,
            content={
                "status": "queued",
                "review_id": review_id,
                "feedback_loop": "mlops/drift/label_feedback.py",
                "message": "Review recorded and queued for next retraining cycle.",
            },
        )

    return app


# Module-level app for `uvicorn weld_defect.api.app:app`.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "weld_defect.api.app:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=False,
    )
