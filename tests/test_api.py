"""API tests — FastAPI TestClient. Health/info work without torch; prediction
requires a model checkpoint (skipped if torch or a checkpoint is unavailable)."""
from __future__ import annotations

import pytest

from fastapi.testclient import TestClient

from weld_defect.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"].startswith("sate-mazandaran-gas")
    assert body["decision_support_only"] is True


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "device" in body


def test_openapi_docs_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert "/predict" in spec["paths"]
    assert "/health" in spec["paths"]


def test_predict_without_model_returns_503(client, tmp_path, png_image):
    # If a checkpoint happens to exist at the default path, this test is N/A.
    import os
    if os.path.exists(os.environ.get("MODEL_CHECKPOINT", "models/ensemble_best.pt")):
        pytest.skip("a model checkpoint is present; 503 path not applicable")
    with open(png_image, "rb") as fh:
        r = client.post("/predict", files={"file": ("x.png", fh, "image/png")})
    assert r.status_code in (503, 422)  # 503 if model missing, 422 if validation


def test_predict_with_model(client, tmp_path, png_image):
    """Full prediction path — requires torch."""
    torch = pytest.importorskip("torch")
    from weld_defect.models.factory import save_model
    from weld_defect.models import CustomCNN
    import os

    ckpt = tmp_path / "m.pt"
    m = CustomCNN(num_classes=6, input_size=224)
    save_model(m, ckpt, meta={"architecture": "custom_cnn", "num_classes": 6,
                              "input_size": 224, "class_names":
                              ["no_defect","porosity","crack","lack_of_fusion","undercut","slag_inclusion"]})
    os.environ["MODEL_CHECKPOINT"] = str(ckpt)
    # Reset the cached model so the new checkpoint is picked up
    import weld_defect.api.app as appmod
    appmod._MODEL = None
    appmod._MODEL_META = None
    with open(png_image, "rb") as fh:
        r = client.post("/predict", files={"file": ("x.png", fh, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision_support_only"] is True
    assert "predicted_class" in body and "top_k" in body
    assert 0.0 <= body["confidence"] <= 1.0
