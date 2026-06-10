from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from .config import DEFAULT_MODELS, DEFAULT_SEED, ID_COL, TARGET_COL
from .io_utils import (
    ensure_dir,
    read_wide_csv,
    resolve_file,
    save_json,
    set_global_seed,
    setup_logger,
    validate_seed,
    validate_train_test,
)
from .metrics import inverse_log_target, log_target, regression_metrics
from .modeling import extract_feature_importance, extract_training_history, fit_model, make_model
from .plots import plot_training_outputs
from .preprocess import FeatureProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--out-dir", default="outputs/full_run")
    parser.add_argument("--id-col", default=ID_COL)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--pca-dims", nargs="*", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--drop-constant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-duplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-row-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--predict-test-from-folds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--no-test-predict", action="store_true")
    return parser.parse_args()


def make_folds(y_log: np.ndarray, n_splits: int, seed: int):
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    try:
        n_bins = min(10, max(2, n_splits * 2))
        bins = pd.qcut(y_log, q=n_bins, labels=False, duplicates="drop")
        if pd.Series(bins).nunique(dropna=True) >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return list(splitter.split(np.zeros_like(y_log), bins))
    except Exception:
        pass

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros_like(y_log)))




def make_log_clip_bounds(y_log_reference: np.ndarray) -> tuple[float, float, float]:
    """Return robust lower/upper/median bounds for log-space predictions.

    The bounds are estimated from the training target of the current fold.
    This prevents extremely large model outputs from overflowing inside expm1().
    """
    ref = np.asarray(y_log_reference, dtype=np.float64)
    ref = ref[np.isfinite(ref)]

    if ref.size == 0:
        return 0.0, 30.0, 0.0

    log_min = float(np.nanpercentile(ref, 0.1) - 1.0)
    log_max = float(np.nanpercentile(ref, 99.9) + 1.0)
    log_med = float(np.nanmedian(ref))

    if not np.isfinite(log_min):
        log_min = 0.0
    if not np.isfinite(log_max):
        log_max = 30.0
    if not np.isfinite(log_med):
        log_med = 0.0

    # np.expm1 overflows around 709 for float64. Keep a safety margin.
    log_min = max(log_min, -700.0)
    log_max = min(log_max, 700.0)

    if log_min > log_max:
        log_min, log_max = log_max, log_min

    return log_min, log_max, log_med


def sanitize_log_predictions(
    y_log_pred: np.ndarray,
    y_log_reference: np.ndarray,
    logger=None,
    context: str = "",
) -> np.ndarray:
    """Replace non-finite log predictions and clip outliers before inverse_log_target()."""
    log_min, log_max, log_med = make_log_clip_bounds(y_log_reference)

    pred = np.asarray(y_log_pred, dtype=np.float64)
    nonfinite_count = int((~np.isfinite(pred)).sum())
    finite_mask = np.isfinite(pred)
    clipped_count = int(((pred[finite_mask] < log_min) | (pred[finite_mask] > log_max)).sum())

    pred = np.nan_to_num(pred, nan=log_med, posinf=log_max, neginf=log_min)
    pred = np.clip(pred, log_min, log_max)

    if logger is not None and (nonfinite_count or clipped_count):
        logger.warning(
            "%s sanitized log predictions: nonfinite=%s, clipped=%s, bounds=[%.6g, %.6g]",
            context,
            nonfinite_count,
            clipped_count,
            log_min,
            log_max,
        )

    return pred

def model_metric_row(model_name, fold, split, y_true, pred):
    row = {"model": model_name, "fold": int(fold), "split": split}
    row.update(regression_metrics(y_true, pred))
    return row


def save_model_artifact(model, processor, out_dir: Path, model_name: str, fold: int):
    model_dir = ensure_dir(out_dir / "models" / model_name)
    joblib.dump(
        {"model": model, "processor": processor},
        model_dir / f"fold_{fold}.joblib",
        compress=3,
    )


def train_cv_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: list[str],
    args,
    folds,
    logger,
):
    y = train_df[args.target_col].astype(float).to_numpy()
    y_log = log_target(y)
    oof_log = np.zeros(len(train_df), dtype=np.float64)
    test_log_folds = []
    metric_rows = []
    history_rows = []
    importance_rows = []
    processor_summaries = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        logger.info("[%s] fold %s/%s", model_name, fold, len(folds))
        train_part = train_df.iloc[train_idx]
        valid_part = train_df.iloc[valid_idx]

        processor = FeatureProcessor(
            drop_constant=args.drop_constant,
            drop_duplicate=args.drop_duplicate,
            add_row_stats=args.add_row_stats,
        )
        X_train = processor.fit_transform(train_part, feature_columns=feature_cols)
        X_valid = processor.transform(valid_part)
        processor_summaries.append({"model": model_name, "fold": fold, **processor.summary()})

        model = make_model(
            model_name,
            seed=args.seed,
            n_features=X_train.shape[1],
            n_samples=X_train.shape[0],
            svd_components=args.svd_components,
        )
        fit_model(
            model,
            model_name,
            X_train,
            y_log[train_idx],
            X_valid,
            y_log[valid_idx],
            early_stopping_rounds=args.early_stopping_rounds,
        )

        fold_y_log = y_log[train_idx]
        valid_log = sanitize_log_predictions(
            model.predict(X_valid),
            fold_y_log,
            logger=logger,
            context=f"[{model_name}] fold {fold} valid",
        )
        train_log = sanitize_log_predictions(
            model.predict(X_train),
            fold_y_log,
            logger=logger,
            context=f"[{model_name}] fold {fold} train",
        )
        oof_log[valid_idx] = valid_log

        valid_pred = inverse_log_target(valid_log)
        train_pred = inverse_log_target(train_log)
        metric_rows.append(model_metric_row(model_name, fold, "train", y[train_idx], train_pred))
        metric_rows.append(model_metric_row(model_name, fold, "valid", y[valid_idx], valid_pred))

        for row in extract_training_history(model, model_name):
            row.update({"model": model_name, "fold": fold})
            history_rows.append(row)

        importance = extract_feature_importance(model, processor.output_feature_names_)
        if importance is not None:
            for feature, value in zip(processor.output_feature_names_, importance):
                importance_rows.append(
                    {"model": model_name, "fold": fold, "feature": feature, "importance": float(value)}
                )

        if test_df is not None and args.predict_test_from_folds:
            X_test = processor.transform(test_df)
            test_log = sanitize_log_predictions(
                model.predict(X_test),
                fold_y_log,
                logger=logger,
                context=f"[{model_name}] fold {fold} test",
            )
            test_log_folds.append(test_log)

        if args.save_models:
            save_model_artifact(model, processor, Path(args.out_dir), model_name, fold)

        del X_train, X_valid, model
        gc.collect()

    oof_log = sanitize_log_predictions(
        oof_log,
        y_log,
        logger=logger,
        context=f"[{model_name}] oof",
    )
    oof_pred = inverse_log_target(oof_log)

    test_pred = None
    if test_df is not None and test_log_folds:
        test_log_mean = sanitize_log_predictions(
            np.mean(np.vstack(test_log_folds), axis=0),
            y_log,
            logger=logger,
            context=f"[{model_name}] test mean",
        )
        test_pred = inverse_log_target(test_log_mean)

    return {
        "model": model_name,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "metrics": metric_rows,
        "history": history_rows,
        "importance": importance_rows,
        "processor_summaries": processor_summaries,
    }


def run_pca_sweep(train_df, feature_cols, args, folds, logger) -> list[dict]:
    if not args.pca_dims:
        return []

    from sklearn.decomposition import TruncatedSVD
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = train_df[args.target_col].astype(float).to_numpy()
    y_log = log_target(y)
    rows = []

    for n_components in args.pca_dims:
        logger.info("[pca_sweep] n_components=%s", n_components)
        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            train_part = train_df.iloc[train_idx]
            valid_part = train_df.iloc[valid_idx]
            processor = FeatureProcessor(
                drop_constant=args.drop_constant,
                drop_duplicate=args.drop_duplicate,
                add_row_stats=args.add_row_stats,
            )
            X_train = processor.fit_transform(train_part, feature_columns=feature_cols)
            X_valid = processor.transform(valid_part)
            safe_components = max(2, min(int(n_components), X_train.shape[1] - 1, X_train.shape[0] - 1))

            model = Pipeline(
                steps=[
                    ("scale", StandardScaler(with_mean=False)),
                    ("svd", TruncatedSVD(n_components=safe_components, random_state=args.seed)),
                    ("ridge", Ridge(alpha=10.0)),
                ]
            )
            model.fit(X_train, y_log[train_idx])

            fold_y_log = y_log[train_idx]
            train_log = sanitize_log_predictions(
                model.predict(X_train),
                fold_y_log,
                logger=logger,
                context=f"[pca_sweep] n_components={safe_components} fold {fold} train",
            )
            valid_log = sanitize_log_predictions(
                model.predict(X_valid),
                fold_y_log,
                logger=logger,
                context=f"[pca_sweep] n_components={safe_components} fold {fold} valid",
            )

            train_pred = inverse_log_target(train_log)
            valid_pred = inverse_log_target(valid_log)

            for split, idx, pred in [
                ("train", train_idx, train_pred),
                ("valid", valid_idx, valid_pred),
            ]:
                row = {
                    "model": "ridge_svd_sweep",
                    "n_components": int(safe_components),
                    "fold": int(fold),
                    "split": split,
                }
                row.update(regression_metrics(y[idx], pred))
                rows.append(row)

            del X_train, X_valid, model
            gc.collect()

    return rows


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in ["rmsle", "rmse", "mae", "r2"] if c in metrics_df.columns]
    summary_rows = []
    for (model, split), part in metrics_df.groupby(["model", "split"]):
        for metric in metric_cols:
            summary_rows.append(
                {
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "mean": float(part[metric].mean()),
                    "std": float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0,
                    "min": float(part[metric].min()),
                    "max": float(part[metric].max()),
                }
            )
    return pd.DataFrame(summary_rows)


def make_blend(oof_df, test_pred_df, metrics_summary, id_col, target_col, logger):
    valid_rmsle = metrics_summary[
        (metrics_summary["split"] == "valid") & (metrics_summary["metric"] == "rmsle")
    ].copy()
    if valid_rmsle.empty:
        return None, None, {}

    candidates = []
    for model in valid_rmsle["model"].tolist():
        if model == "mean_log":
            continue
        pred_col = f"pred_{model}"
        if pred_col in oof_df.columns and (test_pred_df is None or pred_col in test_pred_df.columns):
            candidates.append(model)

    if not candidates:
        return None, None, {}

    valid_rmsle = valid_rmsle[valid_rmsle["model"].isin(candidates)]
    inv = 1.0 / np.maximum(valid_rmsle["mean"].to_numpy(dtype=float), 1e-9) ** 2
    weights = inv / inv.sum()
    weight_map = {model: float(weight) for model, weight in zip(valid_rmsle["model"], weights)}
    logger.info("blend weights: %s", weight_map)

    oof_blend = np.zeros(len(oof_df), dtype=np.float64)
    for model, weight in weight_map.items():
        oof_blend += weight * oof_df[f"pred_{model}"].to_numpy(dtype=float)

    test_blend = None
    if test_pred_df is not None:
        test_blend = np.zeros(len(test_pred_df), dtype=np.float64)
        for model, weight in weight_map.items():
            test_blend += weight * test_pred_df[f"pred_{model}"].to_numpy(dtype=float)

    return oof_blend, test_blend, weight_map


def run_training(args):
    args.seed = validate_seed(args.seed)
    set_global_seed(args.seed)

    out_dir = ensure_dir(args.out_dir)
    logger = setup_logger(out_dir / "train.log")
    save_json(vars(args), out_dir / "run_config.json")

    data_dir = Path(args.data_dir)
    train_path = resolve_file(data_dir, args.train_file, fallback_names=("preview_train.csv", "数据预览.txt"))
    test_path = resolve_file(data_dir, args.test_file)
    sample_path = resolve_file(data_dir, args.sample_submission)

    logger.info("reading train data from %s", train_path)
    train_df = read_wide_csv(train_path, id_col=args.id_col, target_col=args.target_col, nrows=args.nrows)
    test_df = None

    if not args.no_test_predict and test_path.exists():
        logger.info("reading test data from %s", test_path)
        test_df = read_wide_csv(test_path, id_col=args.id_col, target_col=args.target_col, nrows=args.nrows)
    else:
        logger.info("test prediction disabled or test file not found")

    feature_cols = validate_train_test(train_df, test_df, id_col=args.id_col, target_col=args.target_col)
    logger.info("train shape=%s, feature_count=%s", train_df.shape, len(feature_cols))
    if test_df is not None:
        logger.info("test shape=%s", test_df.shape)

    y = train_df[args.target_col].astype(float).to_numpy()
    folds = make_folds(log_target(y), args.n_splits, args.seed)

    oof_df = train_df[[args.id_col, args.target_col]].copy()
    test_pred_df = test_df[[args.id_col]].copy() if test_df is not None else None

    all_metrics = []
    all_history = []
    all_importance = []
    all_processor_summaries = []

    for model_name in args.models:
        result = train_cv_model(model_name, train_df, test_df, feature_cols, args, folds, logger)
        oof_df[f"pred_{model_name}"] = result["oof_pred"]
        if test_pred_df is not None and result["test_pred"] is not None:
            test_pred_df[f"pred_{model_name}"] = result["test_pred"]
        all_metrics.extend(result["metrics"])
        all_history.extend(result["history"])
        all_importance.extend(result["importance"])
        all_processor_summaries.extend(result["processor_summaries"])

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(out_dir / "metrics_cv.csv", index=False)

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    pca_rows = run_pca_sweep(train_df, feature_cols, args, folds, logger)
    if pca_rows:
        pd.DataFrame(pca_rows).to_csv(out_dir / "pca_sweep_metrics.csv", index=False)

    if all_history:
        pd.DataFrame(all_history).to_csv(out_dir / "training_history.csv", index=False)
    if all_importance:
        pd.DataFrame(all_importance).to_csv(out_dir / "feature_importance.csv", index=False)
    if all_processor_summaries:
        pd.DataFrame(all_processor_summaries).to_csv(out_dir / "processor_summaries.csv", index=False)

    oof_blend, test_blend, weight_map = make_blend(
        oof_df,
        test_pred_df,
        summary_df,
        id_col=args.id_col,
        target_col=args.target_col,
        logger=logger,
    )
    save_json(weight_map, out_dir / "blend_weights.json")

    if oof_blend is not None:
        oof_df["pred_blend"] = oof_blend
        blend_metrics = regression_metrics(oof_df[args.target_col].to_numpy(dtype=float), oof_blend)
        blend_rows = []
        for metric, value in blend_metrics.items():
            blend_rows.append(
                {
                    "model": "blend",
                    "split": "valid",
                    "metric": metric,
                    "mean": float(value),
                    "std": 0.0,
                    "min": float(value),
                    "max": float(value),
                }
            )
        summary_df = pd.concat([summary_df, pd.DataFrame(blend_rows)], ignore_index=True)
        summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)

    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    if test_pred_df is not None:
        if test_blend is not None:
            test_pred_df["pred_blend"] = test_blend
        test_pred_df.to_csv(out_dir / "predictions_test_by_model.csv", index=False)

        if sample_path.exists():
            sample = pd.read_csv(sample_path)
            submission = sample[[args.id_col]].copy()
        else:
            submission = test_df[[args.id_col]].copy()

        pred_col = "pred_blend" if "pred_blend" in test_pred_df.columns else test_pred_df.columns[-1]
        submission = submission.merge(
            test_pred_df[[args.id_col, pred_col]].rename(columns={pred_col: args.target_col}),
            on=args.id_col,
            how="left",
        )
        if submission[args.target_col].isna().any():
            fallback = float(np.expm1(np.mean(log_target(y))))
            submission[args.target_col] = submission[args.target_col].fillna(fallback)
        submission[args.target_col] = np.clip(submission[args.target_col].astype(float), 0.0, None)
        submission.to_csv(out_dir / "submission.csv", index=False)
        logger.info("submission written to %s", out_dir / "submission.csv")

    plot_training_outputs(out_dir)
    logger.info("training complete: %s", out_dir)


if __name__ == "__main__":
    run_training(parse_args())
