"""Unit tests for the pre-launch feature parsers (no data/models needed)."""

import math

import numpy as np

from nocarz.features import (
    PARIS_CENTER,
    compute_distance_features,
    count_premium_amenities,
    parse_bathrooms,
)


def test_parse_bathrooms_numeric_forms():
    assert parse_bathrooms("1 bath") == 1.0
    assert parse_bathrooms("1.5 baths") == 1.5
    assert parse_bathrooms("1 shared bath") == 1.0
    assert parse_bathrooms("0 baths") == 0.0


def test_parse_bathrooms_half_and_missing():
    assert parse_bathrooms("Half-bath") == 0.5
    assert parse_bathrooms("Shared half-bath") == 0.5
    assert math.isnan(parse_bathrooms(None))
    assert math.isnan(parse_bathrooms(float("nan")))


def test_count_premium_amenities():
    am = '["Wifi", "Kitchen", "Air conditioning", "Elevator", "Dishwasher", "Pool"]'
    # 4 premium categories: air conditioning, elevator, dishwasher, pool.
    assert count_premium_amenities(am) == 4
    assert count_premium_amenities("[]") == 0
    assert count_premium_amenities(None) == 0


def test_count_premium_amenities_no_double_count():
    # Two TV variants are not premium and must not inflate the count.
    am = '["TV", "TV with standard cable", "Wifi"]'
    assert count_premium_amenities(am) == 0


def test_distance_features_center_is_zero():
    d = compute_distance_features(*PARIS_CENTER)
    assert float(d["dist_center_km"]) < 1e-6
    # Notre-Dame is a landmark co-located with Point Zéro -> nearest ~ 0.
    assert float(d["dist_nearest_landmark_km"]) < 1e-6


def test_distance_features_bulk_matches_pointwise():
    """Train/serve parity: the bulk (array) path must equal the per-point path."""
    lats = np.array([48.86, 48.88, 48.84])
    lons = np.array([2.34, 2.30, 2.39])
    bulk = compute_distance_features(lats, lons)
    for i in range(len(lats)):
        point = compute_distance_features(lats[i], lons[i])
        assert np.isclose(bulk["dist_center_km"][i], float(point["dist_center_km"]))
        assert np.isclose(
            bulk["dist_nearest_landmark_km"][i], float(point["dist_nearest_landmark_km"])
        )
