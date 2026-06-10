from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cmd = [
    sys.executable,
    "-m",
    "src.train",
    "--data-dir",
    str(ROOT / "sample_data"),
    "--train-file",
    "preview_train.csv",
    "--out-dir",
    str(ROOT / "outputs" / "smoke_test"),
    "--models",
    "mean_log",
    "ridge_svd",
    "--n-splits",
    "2",
    "--seed",
    "114",
    "--svd-components",
    "8",
    "--pca-dims",
    "8",
    "--no-test-predict",
]
print(" ".join(cmd))
subprocess.run(cmd, cwd=ROOT, check=True)
print("Smoke test finished:", ROOT / "outputs" / "smoke_test")
