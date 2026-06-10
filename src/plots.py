from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook")
BASE_COLOR = sns.color_palette()[0]

from .io_utils import ensure_dir


def _savefig(path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _rotate_xticks():
    plt.xticks(rotation=35, ha="right")


def plot_training_outputs(run_dir):
    run_dir = Path(run_dir)
    plot_dir = ensure_dir(run_dir / "plots")

    metrics_path = run_dir / "metrics_cv.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        valid = metrics[metrics["split"] == "valid"].copy()
        if not valid.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=valid, x="model", y="rmsle", hue="model", legend=False)
            _rotate_xticks()
            plt.title("Cross-validation RMSLE by model")
            plt.ylabel("RMSLE")
            _savefig(plot_dir / "cv_rmsle_by_model.png")

            metric_cols = [c for c in ["rmsle", "rmse", "mae", "r2"] if c in valid.columns]
            summary = valid.groupby("model", as_index=False)[metric_cols].mean()
            long_df = summary.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="value")
            plt.figure(figsize=(11, 5))
            sns.barplot(data=long_df, x="model", y="value", hue="metric")
            _rotate_xticks()
            plt.title("Metric comparison on validation folds")
            _savefig(plot_dir / "metric_comparison.png")

            if "train" in set(metrics["split"]):
                train_valid = (
                    metrics.pivot_table(index=["model", "fold"], columns="split", values="rmsle", aggfunc="mean")
                    .reset_index()
                )
                if {"train", "valid"}.issubset(train_valid.columns):
                    train_valid["valid_minus_train"] = train_valid["valid"] - train_valid["train"]
                    plt.figure(figsize=(10, 5))
                    sns.barplot(data=train_valid, x="model", y="valid_minus_train", hue="model", legend=False)
                    _rotate_xticks()
                    plt.title("Overfitting check: validation RMSLE minus training RMSLE")
                    plt.ylabel("RMSLE gap")
                    _savefig(plot_dir / "train_valid_gap.png")

    oof_path = run_dir / "oof_predictions.csv"
    summary_path = run_dir / "metrics_summary.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        pred_cols = [c for c in oof.columns if c.startswith("pred_")]
        best_col = None
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            valid_summary = summary[(summary["split"] == "valid") & (summary["metric"] == "rmsle")]
            if not valid_summary.empty:
                best_model = valid_summary.sort_values("mean").iloc[0]["model"]
                candidate = f"pred_{best_model}"
                if candidate in pred_cols:
                    best_col = candidate
        if best_col is None and pred_cols:
            best_col = pred_cols[0]

        if best_col and "target" in oof.columns:
            sample = oof.sample(n=min(3000, len(oof)), random_state=114514)
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=sample, x="target", y=best_col, s=12, alpha=0.5, color=BASE_COLOR)
            lim = max(sample["target"].max(), sample[best_col].max())
            plt.plot([0, lim], [0, lim], linestyle="--", linewidth=1)
            plt.title(f"OOF actual vs predicted: {best_col.replace('pred_', '')}")
            _savefig(plot_dir / f"oof_actual_vs_pred_{best_col.replace('pred_', '')}.png")

            residual = sample[best_col] - sample["target"]
            plt.figure(figsize=(9, 5))
            sns.histplot(residual, bins=80, kde=True, color=BASE_COLOR)
            plt.title(f"OOF residuals: {best_col.replace('pred_', '')}")
            plt.xlabel("prediction - target")
            _savefig(plot_dir / f"residuals_{best_col.replace('pred_', '')}.png")

    hist_path = run_dir / "training_history.csv"
    if hist_path.exists():
        history = pd.read_csv(hist_path)
        if not history.empty:
            metric_guess = history["metric"].iloc[0]
            subset = history[history["metric"] == metric_guess].copy()
            plt.figure(figsize=(11, 5))
            sns.lineplot(data=subset, x="iteration", y="value", hue="model", units="fold", estimator=None, alpha=0.35)
            plt.title(f"Training history: {metric_guess}")
            _savefig(plot_dir / "training_history.png")

    fi_path = run_dir / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        if not fi.empty:
            agg = fi.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
            top = agg.head(50)
            plt.figure(figsize=(9, max(7, 0.18 * len(top))))
            sns.barplot(data=top, y="feature", x="importance", hue="feature", legend=False)
            plt.title("Top 50 mean feature importances")
            _savefig(plot_dir / "feature_importance_top50.png")

    pca_path = run_dir / "pca_sweep_metrics.csv"
    if pca_path.exists():
        pca = pd.read_csv(pca_path)
        if not pca.empty and "n_components" in pca.columns:
            plt.figure(figsize=(8, 5))
            sns.lineplot(data=pca[pca["split"] == "valid"], x="n_components", y="rmsle", marker="o", color=BASE_COLOR)
            plt.title("Dimensionality reduction sweep: Ridge + TruncatedSVD")
            plt.ylabel("Validation RMSLE")
            _savefig(plot_dir / "pca_sweep.png")

    pred_path = run_dir / "predictions_test_by_model.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        pred_cols = [c for c in preds.columns if c.startswith("pred_")]
        if pred_cols:
            sample = preds[pred_cols].sample(n=min(10000, len(preds)), random_state=114514)
            long_pred = sample.melt(var_name="model", value_name="prediction")
            plt.figure(figsize=(10, 5))
            sns.histplot(data=long_pred, x="prediction", hue="model", bins=80, element="step", common_norm=False)
            plt.title("Test prediction distribution by model")
            _savefig(plot_dir / "prediction_distribution.png")
