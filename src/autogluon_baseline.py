from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL
from .io_utils import ensure_dir, read_wide_csv, resolve_file, setup_logger, validate_train_test
from .metrics import log_target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-file", default="train.csv")
    parser.add_argument("--test-file", default="test.csv")
    parser.add_argument("--sample-submission", default="sample_submission.csv")
    parser.add_argument("--out-dir", default="outputs/autogluon_run")
    parser.add_argument("--time-limit", type=int, default=14400)
    parser.add_argument("--id-col", default=ID_COL)
    parser.add_argument("--target-col", default=TARGET_COL)
    return parser.parse_args()


def main(args):
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise ImportError("AutoGluon is not installed. Run: pip install -r requirements-extra.txt") from exc

    out_dir = ensure_dir(args.out_dir)
    logger = setup_logger(out_dir / "autogluon.log")
    data_dir = Path(args.data_dir)
    train_path = resolve_file(data_dir, args.train_file)
    test_path = resolve_file(data_dir, args.test_file)
    sample_path = resolve_file(data_dir, args.sample_submission)

    train_df = read_wide_csv(train_path, id_col=args.id_col, target_col=args.target_col)
    test_df = read_wide_csv(test_path, id_col=args.id_col, target_col=args.target_col)
    feature_cols = validate_train_test(train_df, test_df, id_col=args.id_col, target_col=args.target_col)

    ag_train = train_df[[args.id_col] + feature_cols].copy()
    ag_train["target_log"] = log_target(train_df[args.target_col].astype(float).to_numpy())
    ag_test = test_df[[args.id_col] + feature_cols].copy()

    predictor = TabularPredictor(
        label="target_log",
        problem_type="regression",
        eval_metric="root_mean_squared_error",
        path=str(out_dir / "autogluon_models"),
    )
    predictor.fit(
        train_data=ag_train.drop(columns=[args.id_col]),
        presets="best_quality",
        time_limit=args.time_limit,
    )

    pred_log = predictor.predict(ag_test.drop(columns=[args.id_col]))
    pred = np.clip(np.expm1(np.asarray(pred_log, dtype=float)), 0.0, None)

    if sample_path.exists():
        sub = pd.read_csv(sample_path)[[args.id_col]].copy()
        pred_df = pd.DataFrame({args.id_col: test_df[args.id_col], args.target_col: pred})
        sub = sub.merge(pred_df, on=args.id_col, how="left")
    else:
        sub = pd.DataFrame({args.id_col: test_df[args.id_col], args.target_col: pred})
    sub[args.target_col] = sub[args.target_col].fillna(float(np.median(pred)))
    sub.to_csv(out_dir / "submission_autogluon.csv", index=False)
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(out_dir / "autogluon_leaderboard.csv", index=False)
    logger.info("AutoGluon artifacts written to %s", out_dir)


if __name__ == "__main__":
    main(parse_args())
