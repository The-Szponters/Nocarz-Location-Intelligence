"""Model registry: load the serialized models + their versions at startup.

registry.json maps role ("a"/"b") -> target ("revenue"/"occupancy") ->
{path, version, type}. Each role serves *both* Canvas outputs with a matched
pair of models. Logging the per-target ``model_version`` per request means a
future model version can be rolled out by dropping a new joblib and bumping
registry.json — no service code change — while old logs remain attributable to
the version that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from nocarz.features import MODELS_DIR, PROJECT_ROOT

# Import so joblib can un-pickle the baseline estimator.
from nocarz.baseline import DistrictMeanRegressor  # noqa: F401

REGISTRY_PATH = MODELS_DIR / "registry.json"
TARGETS = ("revenue", "occupancy")
# Valid output range per target (revenue >= 0; occupancy is a fraction in [0,1]).
_CLIP = {"revenue": (0.0, None), "occupancy": (0.0, 1.0)}


def _clamp(target: str, value: float) -> float:
    lo, hi = _CLIP[target]
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


class ModelRegistry:
    def __init__(self, models: dict, versions: dict, meta: dict):
        self.models = models      # {"a": {"revenue": est, "occupancy": est}, "b": {...}}
        self.versions = versions  # {"a": {"revenue": "...", "occupancy": "..."}, ...}
        self.meta = meta          # full registry.json content

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "ModelRegistry":
        meta = json.loads(Path(path).read_text())
        models, versions = {}, {}
        for role in ("a", "b"):
            models[role], versions[role] = {}, {}
            for target in TARGETS:
                entry = meta[role][target]
                models[role][target] = joblib.load(PROJECT_ROOT / entry["path"])
                versions[role][target] = entry["version"]
        return cls(models, versions, meta)

    def predict(self, role: str, X) -> dict:
        """Predict both Canvas outputs, clamped to their valid ranges."""
        return {
            target: _clamp(target, float(self.models[role][target].predict(X)[0]))
            for target in TARGETS
        }

    def version(self, role: str, target: str = "revenue") -> str:
        return self.versions[role][target]
