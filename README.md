# Regression CE Research Code

This repository contains a reproducible experiment pipeline for the anonymous wide sparse regression task described in the uploaded assignment. It is designed for the Santander Value Prediction style dataset: `ID`, many anonymized numerical features, and a continuous positive `target`.

## Data layout

Place the full files in `data/` before running the full experiment:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

The package also includes `sample_data/preview_train.csv`, converted from the uploaded preview, for smoke testing only.

## Quick smoke test

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

## Full cloud run

```bash
pip install -r requirements.txt

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
```

## Cloud notebook

Open:

```text
notebooks/cloud_training.ipynb
```

It contains cells for package installation, data upload or mounted drive import, diagnostics, training, visualization, and result packaging.

## Generated materials

A normal run writes:

```text
outputs/full_run/
  submission.csv
  predictions_test_by_model.csv
  oof_predictions.csv
  metrics_cv.csv
  metrics_summary.csv
  pca_sweep_metrics.csv
  feature_importance.csv
  training_history.csv
  blend_weights.json
  run_config.json
  plots/
    cv_rmsle_by_model.png
    metric_comparison.png
    train_valid_gap.png
    oof_actual_vs_pred_*.png
    residuals_*.png
    pca_sweep.png
    feature_importance_top50.png
    prediction_distribution.png
```

These files are intended to support later reporting: metric comparison, overfitting analysis, dimensionality reduction comparison, residual analysis, feature importance, prediction distribution, and training curves.

## Model families included

Default baselines:

- `mean_log`: constant baseline on log-transformed target.
- `ridge_svd`: linear baseline after TruncatedSVD dimensionality reduction.
- `elasticnet_svd`: sparse linear baseline after TruncatedSVD.
- `lightgbm`: gradient boosting tree baseline suitable for sparse high-dimensional tabular data.
- `xgboost`: scalable sparsity-aware tree boosting baseline.
- `catboost`: ordered boosting baseline.

Optional baseline:

- `tabpfn`: tabular foundation model wrapper with fold-local feature selection. Install `requirements-extra.txt` before use.

## References used for implementation choices

- Kaggle Santander Value Prediction Challenge: https://www.kaggle.com/c/santander-value-prediction-challenge
- Public Santander solution example using LightGBM and XGBoost: https://github.com/minjielu/Kaggle-Santander-Value-Prediction-Challenge
- LightGBM paper: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- XGBoost paper: https://arxiv.org/abs/1603.02754
- CatBoost paper: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Tree models vs deep learning on tabular data: https://arxiv.org/abs/2207.08815
- AutoGluon-Tabular paper: https://arxiv.org/abs/2003.06505
- TabPFN repository and current usage notes: https://github.com/PriorLabs/TabPFN

This code does not implement leak-based Santander approaches. The validation and model-selection logic uses only training targets.
