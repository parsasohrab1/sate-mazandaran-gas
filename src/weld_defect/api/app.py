"""FastAPI inference service.

Endpoints:
    GET  /             — service info
    GET  /health       — liveness + model load status
    POST /predict      — single-image prediction (multipart upload)
    POST /predict/batch — batch prediction (multipart upload, ≤ max_upload_mb)

The service loads the model once at startup (env MODEL_REGISTRY or
``models/ensemble_best.pt``) and is stateless across requests.

Safety banner: every response carries ``decision_support_only=True`` and the
OpenAPI description restates the SRS §5.4 rule that outputs must be reviewed by
a certified NDT specialist before any safety-critical decision.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from weld_defect import __version__
from weld_defect.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    PredictionItem,
    PredictionResponse,
)

app = FastAPI(
    title="Sate Mazandaran Gas — Weld Defect Detection API",
    description=(
        "Decision-support inference service for weld-defect classification in "
        "pipeline radiographs. Outputs are decision-support only (پشتیبانی از "
        "کارشناس) and must be reviewed by a certified NDT specialist per "
        "API 1104 / ASME Section V before any safety-critical decision (SRS §5.4)."
    ),
    version=__version__,
    contact={"name": "Petro Pala Toos R&D", "email": "r&d@petropalatoos.ir"},
    license_info={"name": "Proprietary — see LICENSE"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("API_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---- Model singleton (lazy) ----------------------------------------------- #
_MODEL = None
_MODEL_META = None


def _load_model():
    global _MODEL, _MODEL_META
    if _MODEL is not None:
        return _MODEL, _MODEL_META
    ckpt = os.environ.get("MODEL_CHECKPOINT", "models/ensemble_best.pt")
    if not Path(ckpt).exists():
        return None, None
    from weld_defect.models.factory import load_model
    import torch
    _MODEL, _MODEL_META = load_model(ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
    return _MODEL, _MODEL_META


@app.on_event("startup")
async def _startup():
    m, _ = _load_model()
    if m is None:
        app.state.logger_warning = f"checkpoint not found at startup; serving /health as degraded"


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "sate-mazandaran-gas-weld-defect-detection",
        "version": __version__,
        "docs": "/docs",
        "decision_support_only": True,
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    model, _ = _load_model()
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        device=device,
        version=__version__,
    )


def _save_upload(upload: UploadFile) -> Path:
    max_mb = float(os.environ.get("API_MAX_UPLOAD_MB", "10"))
    data = upload.file.read()
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f"upload exceeds {max_mb} MB")
    img = Image.open(io.BytesIO(data))
    if img.mode != "L":
        img = img.convert("L")
    tmp = Path(os.environ.get("DATA_ROOT", "./data")) / "api_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / f"{upload.filename or 'upload'}.png"
    img.save(p)
    return p


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    model, meta = _load_model()
    if model is None:
        raise HTTPException(503, "model not loaded — set MODEL_CHECKPOINT and restart")
    from weld_defect.predict import predict as _predict
    p = _save_upload(file)
    out = _predict(p, model=model, meta=meta)
    return PredictionResponse(
        image=p.name,
        predicted_class=out["predicted_class"],
        confidence=out["confidence"],
        top_k=[PredictionItem(class_name=i["class"], probability=i["probability"]) for i in out["top_k"]],
        inference_ms=out["inference_ms"],
        model=out["model"],
        decision_support_only=True,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
    model, meta = _load_model()
    if model is None:
        raise HTTPException(503, "model not loaded")
    from weld_defect.predict import predict as _predict
    if len(files) > 64:
        raise HTTPException(413, "max 64 files per batch")
    results = []
    total_ms = 0.0
    for f in files:
        p = _save_upload(f)
        out = _predict(p, model=model, meta=meta)
        total_ms += out["inference_ms"]
        results.append(PredictionResponse(
            image=p.name,
            predicted_class=out["predicted_class"],
            confidence=out["confidence"],
            top_k=[PredictionItem(class_name=i["class"], probability=i["probability"]) for i in out["top_k"]],
            inference_ms=out["inference_ms"],
            model=out["model"],
            decision_support_only=True,
        ))
    return BatchPredictionResponse(
        results=results,
        count=len(results),
        avg_inference_ms=total_ms / max(1, len(results)),
    )
