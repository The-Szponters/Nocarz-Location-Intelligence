"""Deterministic A/B model routing.

The assignment is a pure function of a stable key (client_id, else listing_id),
so the same client/listing always lands in the same group across restarts and
processes without storing any state. Changing ``SALT`` reproducibly reshuffles
the split; ``NOCARZ_AB_SPLIT`` (env) tunes the A-fraction for canary/shadow runs.
"""

from __future__ import annotations

import hashlib
import os

SALT = "nocarz-ab-2026"


def _split() -> float:
    """Fraction of traffic routed to model A (default 0.5)."""
    try:
        return float(os.environ.get("NOCARZ_AB_SPLIT", "0.5"))
    except ValueError:
        return 0.5


def assign_group(key: str, salt: str = SALT, split: float | None = None) -> str:
    """Map a stable key to 'a' or 'b' via a uniform hash bucket."""
    if split is None:
        split = _split()
    digest = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # uniform in [0, 1]
    return "a" if bucket < split else "b"


def resolve_model(client_id: str | None, listing_id: int,
                  force_model: str | None) -> tuple[str, str]:
    """Return (model, reason). force_model wins; else sticky hash routing."""
    if force_model in ("a", "b"):
        return force_model, "forced"
    key = client_id if client_id else str(listing_id)
    return assign_group(key), "hash"
