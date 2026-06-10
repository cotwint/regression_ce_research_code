from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def package_results(run_dir, zip_path):
    run_dir = Path(run_dir)
    zip_path = Path(zip_path)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in run_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(run_dir.parent))
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--zip-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = package_results(args.run_dir, args.zip_path)
    print(out)
