from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALLOWED_SEEDS, ID_COL, TARGET_COL


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(log_path=None, name="regression_ce") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_path is not None:
        ensure_dir(Path(log_path).parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def validate_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed must be one of {ALLOWED_SEEDS}; got {seed}")
    return seed


def set_global_seed(seed: int) -> None:
    seed = validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_wide_csv(path, id_col=ID_COL, target_col=TARGET_COL, nrows=None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    header = pd.read_csv(path, nrows=0)
    dtype = {}
    for col in header.columns:
        if col == id_col:
            dtype[col] = "string"
        else:
            dtype[col] = "float32"

    return pd.read_csv(path, dtype=dtype, nrows=nrows, low_memory=False)


def resolve_file(data_dir, filename, fallback_names=()):
    data_dir = Path(data_dir)
    candidates = [data_dir / filename]
    for name in fallback_names:
        candidates.append(data_dir / name)
    for p in candidates:
        if p.exists():
            return p
    return data_dir / filename


def validate_train_test(train_df, test_df=None, id_col=ID_COL, target_col=TARGET_COL):
    if id_col not in train_df.columns:
        raise ValueError(f"training data must contain id column {id_col!r}")
    if target_col not in train_df.columns:
        raise ValueError(f"training data must contain target column {target_col!r}")

    feature_cols = [c for c in train_df.columns if c not in (id_col, target_col)]
    if not feature_cols:
        raise ValueError("no feature columns found")

    if test_df is not None:
        if id_col not in test_df.columns:
            raise ValueError(f"test data must contain id column {id_col!r}")
        missing = [c for c in feature_cols if c not in test_df.columns]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"test data is missing {len(missing)} training features; examples: {preview}")

    return feature_cols


def write_dataframe(df, path, index=False):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
