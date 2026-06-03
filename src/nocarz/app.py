"""Nocarz revenue-prediction microservice (FastAPI).

Endpoints
---------
POST /predict_revenue          main endpoint; A/B model choice is transparent
                               to the client (model identity is NOT returned).
GET  /health                   liveness + loaded model versions.
POST /predict_revenue/{model}  debug/forced endpoint (model in {a,b}); reveals
                               the model. Used by tests and the paired demo.

Both models + the spatial FeatureBuilder are loaded once at startup (lifespan).
Run with: uvicorn nocarz.app:app --host 127.0.0.1 --port 8080 --workers 1
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from nocarz.features import FeatureBuilder
from nocarz.logging_io import SCHEMA_VERSION, append_log
from nocarz.model_registry import ModelRegistry
from nocarz.routing import resolve_model
from nocarz.schemas import (
    DebugPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["registry"] = ModelRegistry.load()
    STATE["feature_builder"] = FeatureBuilder.from_listings_csv()
    yield
    STATE.clear()


app = FastAPI(title="Nocarz Revenue Predictor", version="1.0.0", lifespan=lifespan)


def _predict_and_log(req: PredictionRequest, role: str, reason: str,
                     endpoint: str) -> tuple[str, dict, str]:
    registry: ModelRegistry = STATE["registry"]
    fb: FeatureBuilder = STATE["feature_builder"]
    f = req.features

    t0 = time.perf_counter()
    X = fb.build_serving_row(f.model_dump())
    preds = registry.predict(role, X)  # {"revenue": ..., "occupancy": ...}
    latency_ms = (time.perf_counter() - t0) * 1000.0

    request_id = str(uuid.uuid4())
    derived = {c: float(X.iloc[0][c]) for c in (
        "dist_center_km", "dist_nearest_landmark_km",
        "comp_count_250m", "comp_count_500m", "comp_count_1000m",
        "district_median_price", "district_price_volatility",
        "district_mean_review_location",
    )}
    append_log({
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "assigned_model": role,
        "assignment_reason": reason,
        "model_version": registry.version(role, "revenue"),
        "occupancy_model_version": registry.version(role, "occupancy"),
        "client_id": req.client_id,
        "listing_id": f.listing_id,
        "input_features": f.model_dump(),
        "derived_features": derived,
        "predicted_annual_revenue": preds["revenue"],
        "predicted_occupancy": preds["occupancy"],
        "latency_ms": round(latency_ms, 3),
        "ground_truth_annual_revenue": None,  # filled offline in A/B evaluation
        "ground_truth_occupancy": None,       # filled offline in A/B evaluation
        "schema_version": SCHEMA_VERSION,
    })
    return request_id, preds, registry.version(role, "revenue")


@app.get("/health")
def health() -> dict:
    registry: ModelRegistry = STATE.get("registry")
    if registry is None:
        raise HTTPException(status_code=503, detail="models not loaded")
    return {
        "status": "ok",
        "models": registry.versions,
        "targets": registry.meta.get("targets"),
    }


@app.post("/predict_revenue", response_model=PredictionResponse)
def predict_revenue(req: PredictionRequest) -> PredictionResponse:
    role, reason = resolve_model(req.client_id, req.features.listing_id, req.force_model)
    request_id, preds, _ = _predict_and_log(req, role, reason, "/predict_revenue")
    return PredictionResponse(
        request_id=request_id,
        listing_id=req.features.listing_id,
        predicted_annual_revenue=round(preds["revenue"], 2),
        predicted_occupancy=round(preds["occupancy"], 4),
    )


@app.post("/predict_revenue/{model}", response_model=DebugPredictionResponse)
def predict_revenue_forced(model: str, req: PredictionRequest) -> DebugPredictionResponse:
    if model not in ("a", "b"):
        raise HTTPException(status_code=400, detail="model must be 'a' or 'b'")
    request_id, preds, version = _predict_and_log(
        req, model, "forced", f"/predict_revenue/{model}"
    )
    return DebugPredictionResponse(
        request_id=request_id,
        listing_id=req.features.listing_id,
        predicted_annual_revenue=round(preds["revenue"], 2),
        predicted_occupancy=round(preds["occupancy"], 4),
        model_used=model,
        model_version=version,
    )
