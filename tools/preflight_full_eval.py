"""Preflight checks for later full-model quantization and lm-eval runs.

This script does not quantize or evaluate a model. It checks whether the local
environment has the optional packages, model directories, and quantized output
directories expected by the later AWQ/RN/lm-eval phase.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


PACKAGE_CHECKS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("psutil", "psutil"),
    ("kneed", "kneed"),
    ("lm-eval", "lm_eval"),
    ("accelerate", "accelerate"),
    ("sentencepiece", "sentencepiece"),
]

REQUIRED_REPO_FILES = [
    "rtn_xl.py",
    "awq_stand_xl.py",
    "awq_dh_xl.py",
    "awq_js_xl.py",
    "awq_gqa_xl.py",
    "compare_awq_slicing.py",
    "compare_awq_heuristic.py",
    "SMART_FLIP_REFERENCE.md",
    "MODEL_DOWNLOADS.md",
    "requirements-full-eval.txt",
]


def exists_mark(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def module_mark(module_name: str) -> str:
    return "OK" if importlib.util.find_spec(module_name) is not None else "MISSING"


def print_check(label: str, status: str, detail: str) -> None:
    print(f"[{status:<7}] {label:<34} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check full-model/lm-eval prerequisites")
    parser.add_argument("--model-path", default="./models/Llama-3-8B")
    parser.add_argument("--awq-standard-dir", default="./quantized_models/llama3_awq_standard")
    parser.add_argument("--awq-dh-dir", default="./quantized_models/llama3_awq_dh")
    parser.add_argument("--awq-gqa-dir", default="./quantized_models/llama3_awq_gqa")
    args = parser.parse_args()

    root = Path.cwd()
    model_path = Path(args.model_path)
    output_dirs = [
        ("RN output", Path("./quantized_models/llama3_rtn_w4g128")),
        ("standard AWQ output", Path(args.awq_standard_dir)),
        ("dynamic heuristic AWQ output", Path(args.awq_dh_dir)),
        ("AWQ GQA ReFlip output", Path(args.awq_gqa_dir)),
    ]

    print("=" * 80)
    print("Full-Model / lm-eval Preflight")
    print("=" * 80)
    print(f"Repo:       {root}")
    print(f"Model path: {model_path}")
    print("=" * 80)

    missing_required = []

    print("\n[1] Repo files")
    for rel_path in REQUIRED_REPO_FILES:
        path = root / rel_path
        status = exists_mark(path)
        print_check(rel_path, status, str(path))
        if status != "OK":
            missing_required.append(rel_path)

    print("\n[2] Python packages")
    missing_packages = []
    for package_name, module_name in PACKAGE_CHECKS:
        status = module_mark(module_name)
        print_check(package_name, status, f"module: {module_name}")
        if status != "OK":
            missing_packages.append(package_name)
            missing_required.append(f"python package: {package_name}")

    if missing_packages:
        print("\nInstall missing later-stage packages with:")
        print("python -m pip install -r requirements-full-eval.txt")

    print("\n[3] Base model")
    model_status = exists_mark(model_path)
    print_check("base model directory", model_status, str(model_path))
    if model_status != "OK":
        missing_required.append(str(model_path))

    print("\n[4] Quantized model outputs")
    missing_outputs = []
    for label, path in output_dirs:
        status = exists_mark(path)
        print_check(label, status, str(path))
        if status != "OK":
            missing_outputs.append((label, path))

    print("\n[5] Next command")
    if model_status != "OK":
        print("Download/place the base model first.")
        print("See MODEL_DOWNLOADS.md for exact model commands.")
    elif missing_outputs:
        print("Generate the first full-model baseline, for example:")
        print(
            f"python awq_stand_xl.py --model-path {args.model_path} "
            f"--output-dir {args.awq_standard_dir} --n-calib 128 --layer-batch-size 16"
        )
    else:
        print("Quantized outputs exist. You can run perplexity comparison or lm-eval next:")
        print(
            f"python compare_awq_slicing.py --heuristic-path {args.awq_dh_dir} "
            f"--standard-path {args.awq_standard_dir} --n-samples 500"
        )
        print(
            "lm_eval --model hf --model_args "
            f"pretrained={args.awq_standard_dir},trust_remote_code=True "
            "--tasks arc_challenge,arc_easy,boolq,piqa,rte --device cuda "
            "--batch_size auto --output_path ./results/lm_eval/llama3_awq_standard"
        )

    if missing_required:
        print("\nRequired items missing before full-model/lm-eval runs:")
        for item in missing_required:
            print(f"- {item}")
        return 1

    if missing_outputs:
        print("\nEnvironment is ready, but quantized model outputs still need to be generated.")
        return 0

    print("\nFull-model/lm-eval prerequisites and expected outputs are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
