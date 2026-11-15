"""FastAPI inference service for weld-defect detection."""
from weld_defect.api.app import app
from weld_defect.api.schemas import (
    HealthResponse,
    PredictionResponse,
    PredictionItem,
    BatchPredictionResponse,
)

__all__ = ["app", "HealthResponse", "PredictionResponse", "PredictionItem", "BatchPredictionResponse"]
