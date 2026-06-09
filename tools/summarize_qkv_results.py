"""Summarize standalone QKV quantization_results.npz.

Run this after `fast_quantize_qkv.py` or `quantize_qkv.py` creates
`quantization_results.npz`. It prints the key comparison metrics and can append
one JSONL row to `results/experiment_ledger.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


REQUIRED_KEYS = [
    "errors_nearest",
    "errors_flip",
    "errors_reflip",
    "rel_errors_nearest",
    "rel_errors_flip",
    "rel_errors_reflip",
    "improvements_h",
    "improvements_r",
    "improvements_hr",
    "attention_scores_orig",
    "attention_scores_nearest",
    "attention_scores_flip",
    "attention_scores_reflip",
]


def as_float(value: object) -> float:
    return float(np.asarray(value).item())


def mean_abs(data: np.ndarray) -> float:
    return float(np.mean(np.abs(data)))


def max_abs(data: np.ndarray) -> float:
    return float(np.max(np.abs(data)))


def mean(data: np.ndarray) -> float:
    return float(np.mean(data))


def summarize(npz_path: Path) -> dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)
    missing = [key for key in REQUIRED_KEYS if key not in data.files]
    if missing:
        raise KeyError(f"Missing expected keys in {npz_path}: {', '.join(missing)}")

    errors_nearest = data["errors_nearest"]
    errors_flip = data["errors_flip"]
    errors_reflip = data["errors_reflip"]
    rel_errors_nearest = data["rel_errors_nearest"]
    rel_errors_flip = data["rel_errors_flip"]
    rel_errors_reflip = data["rel_errors_reflip"]
    improvements_h = data["improvements_h"]
    improvements_r = data["improvements_r"]
    improvements_hr = data["improvements_hr"]

    summary = {
        "result_file": str(npz_path),
        "num_heads": int(len(errors_nearest)),
        "metrics": {
            "nearest_mean_abs_attention_error": mean_abs(errors_nearest),
            "nearest_mean_abs_relative_error_pct": mean_abs(rel_errors_nearest),
            "nearest_max_abs_attention_error": max_abs(errors_nearest),
            "flip_mean_abs_attention_error": mean_abs(errors_flip),
            "flip_mean_abs_relative_error_pct": mean_abs(rel_errors_flip),
            "flip_max_abs_attention_error": max_abs(errors_flip),
            "reflip_mean_abs_attention_error": mean_abs(errors_reflip),
            "reflip_mean_abs_relative_error_pct": mean_abs(rel_errors_reflip),
            "reflip_max_abs_attention_error": max_abs(errors_reflip),
            "nearest_to_flip_mean_improvement_pct": mean(improvements_h),
            "nearest_to_reflip_mean_improvement_pct": mean(improvements_r),
            "flip_to_reflip_mean_improvement_pct": mean(improvements_hr),
            "best_reflip_improvement_pct": as_float(np.max(improvements_r)),
            "best_reflip_head": int(np.argmax(improvements_r)),
            "worst_reflip_improvement_pct": as_float(np.min(improvements_r)),
            "worst_reflip_head": int(np.argmin(improvements_r)),
        },
    }
    return summary


def print_summary(summary: dict[str, object]) -> None:
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)

    print("=" * 80)
    print("Standalone QKV Result Summary")
    print("=" * 80)
    print(f"Result file: {summary['result_file']}")
    print(f"Heads:       {summary['num_heads']}")
    print("\nAttention score error (lower is better):")
    print(f"  Nearest mean abs:  {metrics['nearest_mean_abs_attention_error']:.6f}")
    print(f"  Flip mean abs:     {metrics['flip_mean_abs_attention_error']:.6f}")
    print(f"  ReFlip mean abs:   {metrics['reflip_mean_abs_attention_error']:.6f}")
    print("\nRelative error percent (lower is better):")
    print(f"  Nearest mean abs:  {metrics['nearest_mean_abs_relative_error_pct']:.4f}%")
    print(f"  Flip mean abs:     {metrics['flip_mean_abs_relative_error_pct']:.4f}%")
    print(f"  ReFlip mean abs:   {metrics['reflip_mean_abs_relative_error_pct']:.4f}%")
    print("\nMean improvement percent (higher is better):")
    print(f"  Nearest -> Flip:   {metrics['nearest_to_flip_mean_improvement_pct']:.2f}%")
    print(f"  Nearest -> ReFlip: {metrics['nearest_to_reflip_mean_improvement_pct']:.2f}%")
    print(f"  Flip -> ReFlip:    {metrics['flip_to_reflip_mean_improvement_pct']:.2f}%")
    print("\nReFlip spread:")
    print(
        f"  Best:  {metrics['best_reflip_improvement_pct']:.2f}% "
        f"(head {metrics['best_reflip_head']})"
    )
    print(
        f"  Worst: {metrics['worst_reflip_improvement_pct']:.2f}% "
        f"(head {metrics['worst_reflip_head']})"
    )


def append_ledger(summary: dict[str, object], ledger_path: Path) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "status": "completed",
        "script": "tools/summarize_qkv_results.py",
        "phase": "standalone_qkv_summary",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **summary,
    }
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize quantization_results.npz")
    parser.add_argument("--npz", default="quantization_results.npz", help="Path to result npz")
    parser.add_argument(
        "--append-ledger",
        action="store_true",
        help="Append the summary to results/experiment_ledger.jsonl",
    )
    parser.add_argument("--ledger", default="results/experiment_ledger.jsonl")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"Result file not found: {npz_path}")
        print("Run fast_quantize_qkv.py first, then re-run this summarizer.")
        return 1

    summary = summarize(npz_path)
    print_summary(summary)

    if args.append_ledger:
        ledger_path = Path(args.ledger)
        append_ledger(summary, ledger_path)
        print(f"\nAppended summary to: {ledger_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
