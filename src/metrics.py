from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def clip_nonnegative(values):
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, 0.0, None)


def rmsle(y_true, y_pred) -> float:
    y_true = clip_nonnegative(y_true)
    y_pred = clip_nonnegative(y_pred)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = clip_nonnegative(y_pred)
    metrics = {
        "rmsle": rmsle(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except Exception:
        metrics["r2"] = float("nan")
    return metrics


def inverse_log_target(y_log):
    return clip_nonnegative(np.expm1(np.asarray(y_log, dtype=np.float64)))


def log_target(y):
    return np.log1p(clip_nonnegative(y))
