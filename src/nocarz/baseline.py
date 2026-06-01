"""Baseline model: the simplest sensible spatial predictor.

Predicts the mean annual revenue of the listing's district
(``neighbourhood_cleansed``), falling back to the global mean for districts
unseen during training. This is the "model bazowy (najprostszy możliwy)" from
the task: a pure spatial lookup with no feature interactions.

Kept as a standalone, importable class so it can be pickled by training and
un-pickled by the microservice (joblib needs the class on the import path).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class DistrictMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, group_col: str = "neighbourhood_cleansed"):
        self.group_col = group_col

    def fit(self, X, y):
        df = pd.DataFrame(X).copy()
        df["_y"] = np.asarray(y, dtype=float)
        self.global_mean_ = float(df["_y"].mean())
        self.group_means_ = df.groupby(self.group_col)["_y"].mean().to_dict()
        return self

    def predict(self, X):
        df = pd.DataFrame(X)
        preds = df[self.group_col].map(self.group_means_).fillna(self.global_mean_)
        return preds.to_numpy(dtype=float)
