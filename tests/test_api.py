"""Tests for the FastAPI API."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from weld_defect.api.app import create_app


@pytest.fixture
def client(tiny_checkpoint):
    """TestClient with a tiny model checkpoint."""
    import os
    os.environ["MODEL_CHECKPOINT"] = str(tiny_checkpoint)
    os.environ["MODEL_ARCHITECTURE"] = "custom_cnn"
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """PNG bytes of a small test image."""
    import cv2
    img = np.random.RandomState(0).randint(0, 255, (64, 64), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_get_root(client):
    """GET / returns service info."""
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "sate-mazandaran-gas-weld-defect-detection"
    assert data["decision_support_only"] is True
    assert "classes" in data
    assert len(data["classes"]) == 6


def test_get_health(client):
    """GET /health returns status and model_loaded."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ok", "degraded")
    assert "model_loaded" in data
    assert "device" in data
    assert "version" in data


def test_post_predict(client, sample_image_bytes):
    """POST /predict returns a valid prediction."""
    r = client.post(
        "/predict",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["image"] == "test.png"
    assert data["predicted_class"] in ["no_defect", "porosity", "crack", "lack_of_fusion", "undercut", "slag_inclusion"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_k"]) == 6
    assert data["decision_support_only"] is True
    assert data["inference_ms"] > 0


def test_post_predict_missing_file(client):
    """POST /predict without file returns 422."""
    r = client.post("/predict")
    assert r.status_code == 422


def test_post_predict_batch(client, sample_image_bytes):
    """POST /predict/batch returns results for multiple files."""
    r = client.post(
        "/predict/batch",
        files=[
            ("files", ("a.png", sample_image_bytes, "image/png")),
            ("files", ("b.png", sample_image_bytes, "image/png")),
        ],
    )
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 2
    assert data["ok"] == 2
    assert "avg_inference_ms" in data
    assert data["decision_support_only"] is True


def test_post_review_valid(client):
    """POST /review with valid body returns 201."""
    r = client.post(
        "/review",
        json={
            "image": "test.png",
            "predicted_class": "porosity",
            "confidence": 0.8,
            "status": "approved",
            "reason": "visible pores",
            "reviewer": "expert",
        },
    )
    assert r.status_code == 201
    data = r.json()
    assert data["status"] == "queued"
    assert "review_id" in data


def test_post_review_invalid_status(client):
    """POST /review with invalid status returns 422."""
    r = client.post(
        "/review",
        json={
            "image": "test.png",
            "predicted_class": "porosity",
            "confidence": 0.8,
            "status": "maybe",
        },
    )
    assert r.status_code == 422


def test_post_review_missing_fields(client):
    """POST /review with missing fields returns 422."""
    r = client.post("/review", json={"image": "test.png"})
    assert r.status_code == 422


def test_docs_endpoint(client):
    """GET /docs returns the OpenAPI Swagger UI."""
    r = client.get("/docs")
    assert r.status_code == 200
