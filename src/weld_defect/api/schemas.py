"""Pydantic v2 schemas for the inference API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Defect class name")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    image: str = Field(..., description="Source image filename")
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[PredictionItem]
    inference_ms: float
    model: str
    decision_support_only: bool = Field(
        True,
        description="Outputs are decision-support only and require NDT specialist review (SRS §5.4).",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: str
    version: str


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    count: int
    avg_inference_ms: float
