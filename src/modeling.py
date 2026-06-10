from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_SEED


class MeanLogRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.value_, dtype=np.float64)


def _safe_svd_components(requested: int, n_features: int, n_samples: int) -> int:
    max_components = max(2, min(int(requested), int(n_features) - 1, int(n_samples) - 1))
    return max_components


def make_model(
    name: str,
    seed: int = DEFAULT_SEED,
    n_features: int | None = None,
    n_samples: int | None = None,
    svd_components: int = 128,
):
    name = name.lower()

    if name == "mean_log":
        return MeanLogRegressor()

    if name in {"ridge_svd", "elasticnet_svd"}:
        if n_features is None or n_samples is None:
            raise ValueError("n_features and n_samples are required for SVD models")
        n_comp = _safe_svd_components(svd_components, n_features, n_samples)
        regressor = Ridge(alpha=10.0) if name == "ridge_svd" else ElasticNet(
            alpha=0.001,
            l1_ratio=0.05,
            max_iter=20000,
            random_state=seed,
            selection="cyclic",
        )
        return Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=False)),
                ("svd", TruncatedSVD(n_components=n_comp, random_state=seed)),
                ("regressor", regressor),
            ]
        )

    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM is not installed. Run: pip install lightgbm") from exc
        return LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            n_estimators=6000,
            learning_rate=0.01,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.30,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )

    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost is not installed. Run: pip install xgboost") from exc
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=6000,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.30,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
        )

    if name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost is not installed. Run: pip install catboost") from exc
        return CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=6000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=seed,
            od_type="Iter",
            od_wait=150,
            verbose=False,
            allow_writing_files=False,
        )

    if name == "tabpfn":
        try:
            from .optional_models import TabPFNFeatureSelectorRegressor
        except ImportError as exc:
            raise ImportError("TabPFN extras are not installed. Run: pip install -r requirements-extra.txt") from exc
        return TabPFNFeatureSelectorRegressor(seed=seed, max_features=1024)

    raise ValueError(f"unknown model name: {name}")


def fit_model(model, name: str, X_train, y_train, X_valid=None, y_valid=None, early_stopping_rounds: int = 150):
    name = name.lower()

    if name == "lightgbm" and X_valid is not None:
        import lightgbm as lgb
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=callbacks,
        )
        return model

    if name == "xgboost" and X_valid is not None:
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
            )
        except TypeError:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return model

    if name == "catboost" and X_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, verbose=False)
        return model

    model.fit(X_train, y_train)
    return model


def extract_training_history(model, name: str) -> list[dict]:
    rows = []
    name = name.lower()

    if name == "lightgbm" and hasattr(model, "evals_result_"):
        for dataset, metric_map in model.evals_result_.items():
            for metric, values in metric_map.items():
                for iteration, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "iteration": iteration,
                            "value": float(value),
                        }
                    )
        return rows

    if name == "xgboost" and hasattr(model, "evals_result"):
        try:
            result = model.evals_result()
        except Exception:
            result = {}
        for dataset, metric_map in result.items():
            for metric, values in metric_map.items():
                for iteration, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "iteration": iteration,
                            "value": float(value),
                        }
                    )
        return rows

    if name == "catboost" and hasattr(model, "get_evals_result"):
        try:
            result = model.get_evals_result()
        except Exception:
            result = {}
        for dataset, metric_map in result.items():
            for metric, values in metric_map.items():
                for iteration, value in enumerate(values, start=1):
                    rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "iteration": iteration,
                            "value": float(value),
                        }
                    )
        return rows

    return rows


def extract_feature_importance(model, feature_names: list[str]) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
        if len(values) == len(feature_names):
            return values

    if hasattr(model, "get_feature_importance"):
        try:
            values = np.asarray(model.get_feature_importance(), dtype=np.float64)
            if len(values) == len(feature_names):
                return values
        except Exception:
            return None

    return None
