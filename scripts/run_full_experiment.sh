#!/usr/bin/env bash
set -euo pipefail

python -m src.diagnostics \
  --data-dir data \
  --train-file train.csv \
  --test-file test.csv \
  --out-dir outputs/diagnostics

python -m src.train \
  --data-dir data \
  --train-file train.csv \
  --test-file test.csv \
  --sample-submission sample_submission.csv \
  --out-dir outputs/full_run \
  --models mean_log ridge_svd elasticnet_svd lightgbm xgboost catboost \
  --n-splits 5 \
  --seed 114514 \
  --svd-components 128 \
  --pca-dims 16 32 64 128 256 \
  --save-models

python -m src.package_results \
  --run-dir outputs/full_run \
  --zip-path outputs/full_run_result_pack.zip
