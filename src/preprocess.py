from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


def _as_numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    X = df.loc[:, list(columns)].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0.0)
    return X.astype("float32", copy=False)


def find_constant_columns(X: pd.DataFrame) -> list[str]:
    nunique = X.nunique(dropna=False)
    return nunique.index[nunique <= 1].tolist()


def find_duplicate_columns(X: pd.DataFrame) -> list[str]:
    groups: dict[tuple[str, str, int], list[str]] = {}
    duplicates: list[str] = []

    for col in X.columns:
        series = X[col]
        hashed = pd.util.hash_pandas_object(series, index=False).values
        digest = hashlib.blake2b(hashed.tobytes(), digest_size=16).hexdigest()
        key = (digest, str(series.dtype), len(series))
        is_duplicate = False
        for prior_col in groups.get(key, []):
            if series.equals(X[prior_col]):
                duplicates.append(col)
                is_duplicate = True
                break
        if not is_duplicate:
            groups.setdefault(key, []).append(col)
    return duplicates


def make_row_statistics(X: pd.DataFrame, batch_size: int = 4096) -> pd.DataFrame:
    arr = X.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = arr.shape
    stats = {
        "row_sum": np.empty(n_rows, dtype=np.float32),
        "row_mean": np.empty(n_rows, dtype=np.float32),
        "row_std": np.empty(n_rows, dtype=np.float32),
        "row_min": np.empty(n_rows, dtype=np.float32),
        "row_max": np.empty(n_rows, dtype=np.float32),
        "row_range": np.empty(n_rows, dtype=np.float32),
        "row_nonzero_count": np.empty(n_rows, dtype=np.float32),
        "row_zero_ratio": np.empty(n_rows, dtype=np.float32),
        "row_nonzero_mean": np.empty(n_rows, dtype=np.float32),
        "row_log_sum": np.empty(n_rows, dtype=np.float32),
        "row_log_mean": np.empty(n_rows, dtype=np.float32),
    }

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        block = arr[start:stop]
        nonzero = block != 0.0
        count = nonzero.sum(axis=1).astype(np.float32)
        row_sum = block.sum(axis=1, dtype=np.float64).astype(np.float32)
        row_min = block.min(axis=1)
        row_max = block.max(axis=1)
        clipped = np.clip(block, 0.0, None)
        log_sum = np.log1p(clipped).sum(axis=1, dtype=np.float64).astype(np.float32)

        stats["row_sum"][start:stop] = row_sum
        stats["row_mean"][start:stop] = block.mean(axis=1, dtype=np.float64).astype(np.float32)
        stats["row_std"][start:stop] = block.std(axis=1, dtype=np.float64).astype(np.float32)
        stats["row_min"][start:stop] = row_min
        stats["row_max"][start:stop] = row_max
        stats["row_range"][start:stop] = row_max - row_min
        stats["row_nonzero_count"][start:stop] = count
        stats["row_zero_ratio"][start:stop] = 1.0 - (count / float(n_cols))
        stats["row_nonzero_mean"][start:stop] = np.divide(
            row_sum,
            count,
            out=np.zeros_like(row_sum, dtype=np.float32),
            where=count > 0,
        )
        stats["row_log_sum"][start:stop] = log_sum
        stats["row_log_mean"][start:stop] = log_sum / float(n_cols)

    return pd.DataFrame(stats, index=X.index)


@dataclass
class FeatureProcessor:
    drop_constant: bool = True
    drop_duplicate: bool = True
    add_row_stats: bool = True
    fill_value: float = 0.0
    row_stat_batch_size: int = 4096
    feature_columns_: list[str] = field(default_factory=list)
    keep_columns_: list[str] = field(default_factory=list)
    dropped_constant_: list[str] = field(default_factory=list)
    dropped_duplicate_: list[str] = field(default_factory=list)
    output_feature_names_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, feature_columns: list[str] | None = None):
        if feature_columns is None:
            feature_columns = list(df.columns)
        self.feature_columns_ = list(feature_columns)
        X = _as_numeric_frame(df, self.feature_columns_)

        dropped = set()
        if self.drop_constant:
            self.dropped_constant_ = find_constant_columns(X)
            dropped.update(self.dropped_constant_)

        candidates = [c for c in self.feature_columns_ if c not in dropped]
        if self.drop_duplicate and candidates:
            self.dropped_duplicate_ = find_duplicate_columns(X[candidates])
            dropped.update(self.dropped_duplicate_)

        self.keep_columns_ = [c for c in self.feature_columns_ if c not in dropped]
        self.output_feature_names_ = list(self.keep_columns_)
        if self.add_row_stats:
            self.output_feature_names_.extend(
                [
                    "row_sum",
                    "row_mean",
                    "row_std",
                    "row_min",
                    "row_max",
                    "row_range",
                    "row_nonzero_count",
                    "row_zero_ratio",
                    "row_nonzero_mean",
                    "row_log_sum",
                    "row_log_mean",
                ]
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.keep_columns_ if c not in df.columns]
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(f"input is missing {len(missing)} expected columns; examples: {preview}")

        X = _as_numeric_frame(df, self.keep_columns_)
        if self.add_row_stats:
            stat_df = make_row_statistics(X, batch_size=self.row_stat_batch_size)
            X = pd.concat([X, stat_df], axis=1)
        X = X.loc[:, self.output_feature_names_]
        return X.astype("float32", copy=False)

    def fit_transform(self, df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(df, feature_columns=feature_columns).transform(df)

    def summary(self) -> dict:
        return {
            "n_input_features": len(self.feature_columns_),
            "n_output_features": len(self.output_feature_names_),
            "n_dropped_constant": len(self.dropped_constant_),
            "n_dropped_duplicate": len(self.dropped_duplicate_),
            "dropped_constant_preview": self.dropped_constant_[:20],
            "dropped_duplicate_preview": self.dropped_duplicate_[:20],
            "add_row_stats": self.add_row_stats,
        }
