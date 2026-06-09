"""Preflight checks for the standalone QKV Flip/ReFlip workflow.

This script does not run quantization. It checks whether the local environment
has the files and Python packages needed for the existing standalone scripts.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "numpy",
    "matplotlib",
    "seaborn",
    "pandas",
    "openpyxl",
    "tqdm",
]

REQUIRED_REPO_FILES = [
    "utils_qkv.py",
    "xspot.py",
    "quantize_qkv.py",
    "fast_quantize_qkv.py",
    "visualize_xspot.py",
    "heuristic_verification.py",
    "data10.csv",
]


def exists_mark(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def module_mark(module_name: str) -> str:
    return "OK" if importlib.util.find_spec(module_name) is not None else "MISSING"


def print_check(label: str, status: str, detail: str) -> None:
    print(f"[{status:<7}] {label:<28} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check standalone QKV workflow prerequisites")
    parser.add_argument("--model-path", default="./models/Llama-3-8B")
    parser.add_argument("--xspot-dir", default="./xspot_layer5_group2")
    parser.add_argument("--group-id", type=int, default=2)
    args = parser.parse_args()

    root = Path.cwd()
    model_path = Path(args.model_path)
    xspot_dir = Path(args.xspot_dir)

    missing_required = []

    print("=" * 80)
    print("Standalone QKV Preflight")
    print("=" * 80)
    print(f"Repo:       {root}")
    print(f"Model path: {model_path}")
    print(f"XSpot dir:  {xspot_dir}")
    print("=" * 80)

    print("\n[1] Repo files")
    for rel_path in REQUIRED_REPO_FILES:
        path = root / rel_path
        status = exists_mark(path)
        print_check(rel_path, status, str(path))
        if status != "OK":
            missing_required.append(rel_path)

    print("\n[2] Python packages")
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        status = module_mark(package)
        print_check(package, status, "import check")
        if status != "OK":
            missing_packages.append(package)
            missing_required.append(f"python package: {package}")

    if missing_packages:
        print("\nInstall missing standalone packages with:")
        print("python -m pip install -r requirements-qkv.txt")

    print("\n[3] Model")
    model_status = exists_mark(model_path)
    print_check("model directory", model_status, str(model_path))
    if model_status != "OK":
        missing_required.append(str(model_path))

    print("\n[4] Generated QKV artifacts")
    expected_artifacts = [
        "js_means.npy",
        f"Wq_group{args.group_id}.npy",
        f"Wk_group{args.group_id}.npy",
        f"Wv_group{args.group_id}.npy",
        "metadata.json",
    ]
    missing_artifacts = []
    for filename in expected_artifacts:
        path = xspot_dir / filename
        status = exists_mark(path)
        print_check(filename, status, str(path))
        if status != "OK":
            missing_artifacts.append(filename)

    print("\n[5] Next command")
    if model_status != "OK":
        print("Place or download the model at the model path first.")
        print(f"Expected: {model_path}")
        print("See MODEL_DOWNLOADS.md for example huggingface-cli commands.")
    elif missing_artifacts:
        print("Generate standalone QKV artifacts:")
        print(
            f"python xspot.py --model-path {args.model_path} --layer-id 5 "
            f"--group-id {args.group_id} --n-samples 128 --seqlen 512 "
            f"--output-dir {args.xspot_dir}"
        )
    else:
        print("Run standalone QKV metrics:")
        print(
            "python fast_quantize_qkv.py --critical-dim-pct 0.1 "
            "--knee-tolerance 0.0 --group-size 128 --max-flip-pct 0.1 "
            "--correction-scale 1.0"
        )

    if missing_required:
        print("\nRequired items missing:")
        for item in missing_required:
            print(f"- {item}")
        return 1

    print("\nRequired repo files, packages, and model path are present.")
    if missing_artifacts:
        print("QKV artifacts are not present yet; run xspot.py next.")
    else:
        print("QKV artifacts are present; run fast_quantize_qkv.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
