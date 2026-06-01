from nocarz.routing import assign_group, resolve_model


def test_assignment_is_deterministic():
    assert assign_group("client-123") == assign_group("client-123")


def test_split_is_roughly_balanced():
    groups = [assign_group(f"client-{i}") for i in range(5000)]
    frac_a = groups.count("a") / len(groups)
    assert 0.45 < frac_a < 0.55


def test_split_extremes():
    assert assign_group("x", split=1.0) == "a"
    assert assign_group("x", split=0.0) == "b"


def test_force_model_overrides():
    assert resolve_model("c1", 42, "a") == ("a", "forced")
    assert resolve_model("c1", 42, "b") == ("b", "forced")


def test_sticky_key_falls_back_to_listing_id():
    # No client_id -> keyed by listing_id, still deterministic.
    m1, r1 = resolve_model(None, 999, None)
    m2, r2 = resolve_model(None, 999, None)
    assert (m1, r1) == (m2, r2)
    assert r1 == "hash"
