import pytest
from pydantic import ValidationError

from nocarz.schemas import ListingFeatures, PredictionRequest

VALID = {
    "listing_id": 3109,
    "latitude": 48.85,
    "longitude": 2.35,
    "neighbourhood_cleansed": "Observatoire",
    "property_type": "Entire rental unit",
    "room_type": "Entire home/apt",
    "accommodates": 2,
    "amenities_count": 15,
}


def test_valid_features():
    f = ListingFeatures(**VALID)
    assert f.accommodates == 2


def test_accommodates_must_be_positive():
    bad = dict(VALID, accommodates=0)
    with pytest.raises(ValidationError):
        ListingFeatures(**bad)


def test_latitude_bounds():
    with pytest.raises(ValidationError):
        ListingFeatures(**dict(VALID, latitude=200.0))


def test_force_model_literal():
    req = PredictionRequest(features=VALID, force_model="b")
    assert req.force_model == "b"
    with pytest.raises(ValidationError):
        PredictionRequest(features=VALID, force_model="c")
