# 回归 CE 研究代码

本仓库包含一个可复现的实验流水线，用于已上传作业中描述的匿名宽稀疏回归任务。设计上与 Santander Value Prediction 类型的数据集一致：包含 `ID`、大量匿名数值特征，以及一个连续的正向 `target`。

## 数据布局

在运行完整实验之前，请将完整文件放入 `data/`：

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

该代码包还包含从上传的预览转换得到的 `sample_data/preview_train.csv`，仅用于快速冒烟测试（smoke test）。

## 快速冒烟测试

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

## 完整云端运行

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

## 云端笔记本

打开：

```text
notebooks/cloud_training.ipynb
```

其中包含用于安装依赖、上传/挂载数据、运行诊断、训练、可视化以及打包结果的单元格（cells）。

## 生成的材料

一次正常运行会写出如下内容：

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

这些文件用于后续报告：指标对比、过拟合分析、降维方法对比、残差分析、特征重要性、预测分布与训练曲线等。

## 包含的模型族

默认基线：

- `mean_log`：对目标取对数后的常数基线。
- `ridge_svd`：先用 TruncatedSVD 降维再做线性岭回归基线。
- `elasticnet_svd`：先用 TruncatedSVD 降维再做稀疏线性回归（ElasticNet）。
- `lightgbm`：适用于稀疏高维表格数据的梯度提升树基线。
- `xgboost`：可扩展、支持稀疏输入的提升树基线。
- `catboost`：带有有序提升（ordered boosting）的基线。

可选基线：

- `tabpfn`：带局部折叠特征选择的表格基础模型包装器。使用前请先安装 `requirements-extra.txt` 中的额外依赖。

## 实现选择参考

- Kaggle Santander Value Prediction Challenge: https://www.kaggle.com/c/santander-value-prediction-challenge
- 使用 LightGBM 和 XGBoost 的公共 Santander 解决示例: https://github.com/minjielu/Kaggle-Santander-Value-Prediction-Challenge
- LightGBM 论文: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- XGBoost 论文: https://arxiv.org/abs/1603.02754
- CatBoost 论文: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- 表格数据上树模型与深度学习的比较: https://arxiv.org/abs/2207.08815
- AutoGluon-Tabular 论文: https://arxiv.org/abs/2003.06505
- TabPFN 仓库与当前使用说明: https://github.com/PriorLabs/TabPFN

本代码未实现基于泄露（leak-based）的 Santander 方法。验证与模型选择逻辑仅使用训练标签（targets）。
