"""Model registry: load the serialized models + their versions at startup.

registry.json maps role ("a"/"b") -> {path, version, type}. Logging the
``model_version`` per request means a future model version can be rolled out by
dropping a new joblib and bumping registry.json — no service code change — while
old logs remain attributable to the version that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from nocarz.features import MODELS_DIR, PROJECT_ROOT

# Import so joblib can un-pickle the baseline estimator.
from nocarz.baseline import DistrictMeanRegressor  # noqa: F401

REGISTRY_PATH = MODELS_DIR / "registry.json"


class ModelRegistry:
    def __init__(self, models: dict, versions: dict, meta: dict):
        self.models = models      # {"a": estimator, "b": estimator}
        self.versions = versions  # {"a": "...", "b": "..."}
        self.meta = meta          # full registry.json content

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "ModelRegistry":
        meta = json.loads(Path(path).read_text())
        models, versions = {}, {}
        for role in ("a", "b"):
            entry = meta[role]
            models[role] = joblib.load(PROJECT_ROOT / entry["path"])
            versions[role] = entry["version"]
        return cls(models, versions, meta)

    def predict(self, role: str, X) -> float:
        pred = float(self.models[role].predict(X)[0])
        return max(0.0, pred)  # revenue cannot be negative

    def version(self, role: str) -> str:
        return self.versions[role]
