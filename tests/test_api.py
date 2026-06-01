"""End-to-end API tests via FastAPI TestClient (no running server needed).

These require the trained models (run scripts/train_models.py first); they are
skipped if the artifacts are missing.
"""

import pytest
from fastapi.testclient import TestClient

from nocarz.features import MODELS_DIR
from nocarz.app import app

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "registry.json").exists(),
    reason="models not trained (run scripts/train_models.py)",
)

PAYLOAD = {
    "client_id": "tester-1",
    "features": {
        "listing_id": 3109,
        "latitude": 48.85,
        "longitude": 2.35,
        "neighbourhood_cleansed": "Observatoire",
        "property_type": "Entire rental unit",
        "room_type": "Entire home/apt",
        "accommodates": 2,
        "amenities_count": 15,
    },
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"a", "b"}


def test_predict_returns_prediction_without_model_identity(client):
    r = client.post("/predict_revenue", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_annual_revenue"] >= 0
    assert body["listing_id"] == 3109
    assert "request_id" in body
    # Transparency: client-facing response must NOT reveal the model.
    assert "model_used" not in body and "assigned_model" not in body


def test_forced_endpoints_reveal_model_and_differ(client):
    ra = client.post("/predict_revenue/a", json=PAYLOAD).json()
    rb = client.post("/predict_revenue/b", json=PAYLOAD).json()
    assert ra["model_used"] == "a" and rb["model_used"] == "b"
    # Baseline (district mean) and HGB should give different predictions.
    assert ra["predicted_annual_revenue"] != rb["predicted_annual_revenue"]


def test_invalid_payload_returns_422(client):
    bad = {"features": dict(PAYLOAD["features"], accommodates=0)}
    r = client.post("/predict_revenue", json=bad)
    assert r.status_code == 422


def test_unknown_district_does_not_crash(client):
    payload = {"features": dict(PAYLOAD["features"], neighbourhood_cleansed="Nowhere")}
    r = client.post("/predict_revenue", json=payload)
    assert r.status_code == 200
