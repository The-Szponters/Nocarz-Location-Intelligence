"""Pydantic request/response models for the prediction microservice."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ListingFeatures(BaseModel):
    """Pre-launch attributes of a candidate listing/location.

    The client supplies only these simple fields; the service derives the
    spatial competition + neighbourhood-aggregate features server-side
    (train/serve parity via FeatureBuilder).
    """

    listing_id: int = Field(..., description="Key used to join ground truth in A/B eval")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    neighbourhood_cleansed: str = Field(..., description="Paris district")
    property_type: str
    room_type: str
    accommodates: int = Field(..., ge=1)
    amenities_count: int = Field(..., ge=0)
    bathrooms: Optional[float] = Field(
        None, ge=0, description="Planned bath count; server imputes the city median if omitted"
    )
    premium_amenities_count: Optional[int] = Field(
        None, ge=0, description="Count of premium amenity categories; defaults to 0 if omitted"
    )


class PredictionRequest(BaseModel):
    client_id: Optional[str] = Field(
        None, description="Stable key for deterministic A/B routing (sticky per client)"
    )
    features: ListingFeatures
    force_model: Optional[Literal["a", "b"]] = Field(
        None, description="Test override; bypasses A/B routing when set"
    )


class PredictionResponse(BaseModel):
    """Client-facing response. Model identity is intentionally NOT exposed
    (model choice is transparent to the client, per the task)."""

    request_id: str
    listing_id: int
    predicted_annual_revenue: float
    currency: str = "EUR"


class DebugPredictionResponse(PredictionResponse):
    """Response for the /predict_revenue/{model} debug endpoint (reveals model)."""

    model_used: str
    model_version: str
