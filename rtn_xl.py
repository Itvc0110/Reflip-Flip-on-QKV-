"""Full-model round-to-nearest INT4 baseline.

This is an additive baseline script for comparison with the AWQ/Flip/ReFlip
methods in this repo. It applies the existing nearest group-wise INT4
quantizer from `utils_qkv.py` to every linear layer, stores dequantized weights
back in the original dtype, and saves a normal Hugging Face model directory.

The output is research-friendly FP16/BF16 dequantized weights, not packed INT4.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_qkv import quantize_weight_groupwise_int4


def parse_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def quantize_linear_nearest(
    module: nn.Linear,
    group_size: int,
    chunk_rows: int,
) -> dict[str, Any]:
    weight = module.weight.data
    original_dtype = weight.dtype
    device = weight.device
    out_features = weight.shape[0]

    if chunk_rows <= 0:
        chunk_rows = out_features

    total_numel = 0
    sum_sq_error = 0.0
    sum_abs_error = 0.0
    sum_abs_orig = 0.0
    max_error = 0.0

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        w_chunk = weight[start:end].detach().float().cpu().numpy()
        w_quant, _, _, _ = quantize_weight_groupwise_int4(w_chunk, group_size=group_size)

        diff = w_quant - w_chunk
        total_numel += int(diff.size)
        sum_sq_error += float(np.sum(diff ** 2))
        sum_abs_error += float(np.sum(np.abs(diff)))
        sum_abs_orig += float(np.sum(np.abs(w_chunk)))
        max_error = max(max_error, float(np.max(np.abs(diff))))

        quant_tensor = torch.from_numpy(w_quant).to(device=device, dtype=original_dtype)
        weight[start:end].copy_(quant_tensor)

        del w_chunk, w_quant, diff, quant_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mse = sum_sq_error / max(total_numel, 1)
    mae = sum_abs_error / max(total_numel, 1)
    mean_abs_orig = sum_abs_orig / max(total_numel, 1)
    rel_error_pct = mae / (mean_abs_orig + 1e-10) * 100

    return {
        "numel": total_numel,
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "rel_error_pct": rel_error_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-model RN INT4 baseline")
    parser.add_argument("--model-path", default="./models/Llama-3-8B")
    parser.add_argument("--output-dir", default="./quantized_models/llama3_rtn_w4g128")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--skip-lm-head", action="store_true", help="Do not quantize lm_head")
    args = parser.parse_args()

    dtype = parse_dtype(args.torch_dtype)

    print("=" * 80)
    print("Full-model RN INT4 baseline")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Group size:  {args.group_size}")
    print(f"Chunk rows:  {args.chunk_rows}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not (args.skip_lm_head and name == "lm_head")
    ]

    print(f"Found {len(linear_layers)} linear layers")
    layer_stats: dict[str, Any] = {}

    with torch.no_grad():
        for name, module in tqdm(linear_layers, desc="RN quantizing"):
            stats = quantize_linear_nearest(module, group_size=args.group_size, chunk_rows=args.chunk_rows)
            layer_stats[name] = stats

    total_numel = sum(stats["numel"] for stats in layer_stats.values())
    mean_mse = sum(stats["mse"] * stats["numel"] for stats in layer_stats.values()) / max(total_numel, 1)
    mean_mae = sum(stats["mae"] * stats["numel"] for stats in layer_stats.values()) / max(total_numel, 1)
    max_error = max((stats["max_error"] for stats in layer_stats.values()), default=0.0)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": "rtn_w4_groupwise_asymmetric_dequant",
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "group_size": args.group_size,
        "chunk_rows": args.chunk_rows,
        "skip_lm_head": args.skip_lm_head,
        "total_linear_layers": len(linear_layers),
        "total_numel": total_numel,
        "weighted_mse": mean_mse,
        "weighted_mae": mean_mae,
        "max_error": max_error,
        "layers": layer_stats,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    stats_path = os.path.join(args.output_dir, "rtn_quant_stats.json")
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nSaved model to:", args.output_dir)
    print("Saved stats to:", stats_path)
    print(f"Weighted MAE: {mean_mae:.8f}")
    print(f"Weighted MSE: {mean_mse:.8f}")
    print(f"Max error:    {max_error:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
