from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook")
BASE_COLOR = sns.color_palette()[0]

from .config import ID_COL, TARGET_COL
from .io_utils import ensure_dir, read_wide_csv, resolve_file, save_json, setup_logger, validate_train_test
from .metrics import log_target


def _savefig(path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def compute_feature_profile(train_df: pd.DataFrame, feature_cols: list[str], target_col: str = TARGET_COL) -> pd.DataFrame:
    X = train_df[feature_cols]
    profile = pd.DataFrame(index=feature_cols)
    profile["missing_rate"] = X.isna().mean(axis=0).astype(float)
    profile["zero_rate"] = (X == 0).mean(axis=0).astype(float)
    profile["mean"] = X.mean(axis=0).astype(float)
    profile["std"] = X.std(axis=0).astype(float)
    profile["variance"] = X.var(axis=0).astype(float)
    profile["nunique"] = X.nunique(dropna=False).astype(int)

    y = train_df[target_col].astype(float)
    y_log = log_target(y)
    corr_values = []
    for col in feature_cols:
        try:
            corr = pd.Series(X[col].astype(float)).corr(pd.Series(y_log), method="spearman")
        except Exception:
            corr = np.nan
        corr_values.append(corr)
    profile["spearman_with_log_target"] = corr_values
    profile = profile.reset_index(names="feature")
    return profile


def make_diagnostic_plots(train_df: pd.DataFrame, feature_profile: pd.DataFrame, out_dir: Path):
    plot_dir = ensure_dir(out_dir / "plots")

    target = train_df[TARGET_COL].astype(float)
    plt.figure(figsize=(9, 5))
    sns.histplot(target, bins=80, kde=True, color=BASE_COLOR)
    plt.title("Target distribution")
    _savefig(plot_dir / "target_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(np.log1p(np.clip(target, 0.0, None)), bins=80, kde=True, color=BASE_COLOR)
    plt.title("Log1p target distribution")
    _savefig(plot_dir / "log_target_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(feature_profile["zero_rate"], bins=80, color=BASE_COLOR)
    plt.title("Feature zero-rate distribution")
    plt.xlabel("zero rate")
    _savefig(plot_dir / "feature_zero_rate_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(feature_profile["missing_rate"], bins=80, color=BASE_COLOR)
    plt.title("Feature missing-rate distribution")
    plt.xlabel("missing rate")
    _savefig(plot_dir / "feature_missing_rate_distribution.png")

    plt.figure(figsize=(9, 5))
    sns.histplot(np.log1p(feature_profile["variance"].clip(lower=0.0)), bins=80, color=BASE_COLOR)
    plt.title("Log1p feature variance distribution")
    _savefig(plot_dir / "feature_variance_distribution.png")

    corr_top = feature_profile.reindex(
        feature_profile["spearman_with_log_target"].abs().sort_values(ascending=False).index
    ).head(40)
    if not corr_top.empty:
        plt.figure(figsize=(8, max(6, 0.20 * len(corr_top))))
        sns.barplot(data=corr_top, y="feature", x="spearman_with_log_target", hue="feature", legend=False)
        plt.title("Top Spearman correlations with log target")
        _savefig(plot_dir / "top_target_correlations.png")

        cols = corr_top["feature"].head(25).tolist()
        if len(cols) >= 2:
            corr_mat = train_df[cols].corr(method="spearman")
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_mat, cmap="vlag", center=0.0)
            plt.title("Spearman correlation heatmap among top target-correlated features")
            _savefig(plot_dir / "top_feature_correlation_heatmap.png")


def run_diagnostics(args):
    out_dir = ensure_dir(args.out_dir)
    logger = setup_logger(out_dir / "diagnostics.log")
    train_path = resolve_file(args.data_dir, args.train_file, fallback_names=("preview_train.csv", "数据预览.txt"))
    test_path = resolve_file(args.data_dir, args.test_file)

    logger.info("reading train data from %s", train_path)
    train_df = read_wide_csv(train_path, id_col=args.id_col, target_col=args.target_col, nrows=args.nrows)

    test_df = None
    if test_path.exists():
        logger.info("reading test data from %s", test_path)
        test_df = read_wide_csv(test_path, id_col=args.id_col, target_col=args.target_col, nrows=args.nrows)

    feature_cols = validate_train_test(train_df, test_df, id_col=args.id_col, target_col=args.target_col)
    logger.info("train shape: %s", train_df.shape)
    if test_df is not None:
        logger.info("test shape: %s", test_df.shape)

    profile = compute_feature_profile(train_df, feature_cols, target_col=args.target_col)
    profile.to_csv(out_dir / "feature_profile.csv", index=False)

    y = train_df[args.target_col].astype(float)
    report = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape) if test_df is not None else None,
        "n_features": len(feature_cols),
        "target": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std()),
            "log1p_mean": float(log_target(y).mean()),
            "log1p_std": float(log_target(y).std()),
        },
        "features": {
            "mean_zero_rate": float(profile["zero_rate"].mean()),
            "median_zero_rate": float(profile["zero_rate"].median()),
            "n_constant_features": int((profile["nunique"] <= 1).sum()),
            "n_missing_features": int((profile["missing_rate"] > 0).sum()),
        },
    }
    save_json(report, out_dir / "data_report.json")
    make_diagnostic_plots(train_df, profile, out_dir)
    logger.info("diagnostics written to %s", out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--out-dir", default="outputs/diagnostics")
    parser.add_argument("--id-col", default=ID_COL)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nrows", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
