from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectKBest, f_regression


class TabPFNFeatureSelectorRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, seed=114514, max_features=1024):
        self.seed = seed
        self.max_features = max_features

    def fit(self, X, y):
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise ImportError("TabPFN is not installed. Run: pip install -r requirements-extra.txt") from exc

        k = min(int(self.max_features), X.shape[1])
        self.selector_ = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.selector_.fit_transform(X, y)
        self.model_ = TabPFNRegressor(random_state=self.seed)
        self.model_.fit(X_selected, y)
        return self

    def predict(self, X):
        X_selected = self.selector_.transform(X)
        return np.asarray(self.model_.predict(X_selected), dtype=np.float64)
