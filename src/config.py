from __future__ import annotations

ALLOWED_SEEDS = (114, 514, 1919180, 810, 114514)
DEFAULT_SEED = 114514

ID_COL = "ID"
TARGET_COL = "target"

DEFAULT_MODELS = (
    "mean_log",
    "ridge_svd",
    "elasticnet_svd",
    "lightgbm",
    "xgboost",
    "catboost",
)

TREE_MODELS = {"lightgbm", "xgboost", "catboost", "extratrees"}
LINEAR_MODELS = {"ridge_svd", "elasticnet_svd"}
OPTIONAL_MODELS = {"tabpfn"}
