"""Unit tests for the pre-launch feature parsers (no data/models needed)."""

import math

from nocarz.features import count_premium_amenities, parse_bathrooms


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
