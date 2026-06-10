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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
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


class SafetensorsWeightLoader:
    """Load individual tensors from a local Hugging Face safetensors model."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.weight_map: dict[str, str] = {}
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}

        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            self.weight_map = index.get("weight_map", {})

    def load(self, tensor_name: str) -> torch.Tensor:
        if tensor_name not in self.weight_map:
            raise KeyError(f"{tensor_name!r} not found in model.safetensors.index.json")

        shard_name = self.weight_map[tensor_name]
        if shard_name not in self._shard_cache:
            shard_path = self.model_path / shard_name
            self._shard_cache[shard_name] = load_file(str(shard_path), device="cpu")
        return self._shard_cache[shard_name][tensor_name]

    def clear_cache(self) -> None:
        self._shard_cache.clear()
        gc.collect()


def set_module_parameter(model: nn.Module, parameter_name: str, tensor: torch.Tensor) -> None:
    module_name, _, leaf_name = parameter_name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    setattr(module, leaf_name, nn.Parameter(tensor, requires_grad=False))


def materialize_remaining_meta_parameters(
    model: nn.Module,
    weight_loader: SafetensorsWeightLoader,
) -> int:
    """Replace any remaining meta parameters with real CPU tensors for saving."""
    materialized = 0
    for name, parameter in list(model.named_parameters()):
        if not getattr(parameter, "is_meta", False):
            continue
        tensor = weight_loader.load(name).detach().cpu().to(dtype=parameter.dtype)
        set_module_parameter(model, name, tensor)
        weight_loader.clear_cache()
        materialized += 1
    return materialized


def quantize_linear_nearest(
    module: nn.Linear,
    group_size: int,
    chunk_rows: int,
    weight_loader: SafetensorsWeightLoader | None = None,
    weight_name: str | None = None,
) -> dict[str, Any]:
    weight = module.weight.data
    original_dtype = weight.dtype
    device = weight.device
    out_features = weight.shape[0]
    is_meta = getattr(weight, "is_meta", False)

    source_weight: torch.Tensor | None = None
    if is_meta:
        if weight_loader is None or weight_name is None:
            raise ValueError("Meta tensor encountered without a safetensors fallback loader")
        source_weight = weight_loader.load(weight_name).detach().cpu()
        original_dtype = source_weight.dtype
        device = torch.device("cpu")
        out_features = source_weight.shape[0]

    if chunk_rows <= 0:
        chunk_rows = out_features

    total_numel = 0
    sum_sq_error = 0.0
    sum_abs_error = 0.0
    sum_abs_orig = 0.0
    max_error = 0.0

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        if source_weight is not None:
            w_chunk = source_weight[start:end].float().numpy()
        else:
            w_chunk = weight[start:end].detach().float().cpu().numpy()
        w_quant, _, _, _ = quantize_weight_groupwise_int4(w_chunk, group_size=group_size)

        diff = w_quant - w_chunk
        total_numel += int(diff.size)
        sum_sq_error += float(np.sum(diff ** 2))
        sum_abs_error += float(np.sum(np.abs(diff)))
        sum_abs_orig += float(np.sum(np.abs(w_chunk)))
        max_error = max(max_error, float(np.max(np.abs(diff))))

        quant_tensor = torch.from_numpy(w_quant).to(device=device, dtype=original_dtype)
        if source_weight is not None:
            source_weight[start:end].copy_(quant_tensor.cpu())
        else:
            weight[start:end].copy_(quant_tensor)

        del w_chunk, w_quant, diff, quant_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mse = sum_sq_error / max(total_numel, 1)
    mae = sum_abs_error / max(total_numel, 1)
    mean_abs_orig = sum_abs_orig / max(total_numel, 1)
    rel_error_pct = mae / (mean_abs_orig + 1e-10) * 100

    if source_weight is not None:
        module.weight = nn.Parameter(source_weight.to(dtype=original_dtype), requires_grad=False)

    return {
        "numel": total_numel,
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "rel_error_pct": rel_error_pct,
        "loaded_from_safetensors": source_weight is not None,
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
    weight_loader = SafetensorsWeightLoader(args.model_path)

    print(f"Found {len(linear_layers)} linear layers")
    layer_stats: dict[str, Any] = {}

    with torch.no_grad():
        for name, module in tqdm(linear_layers, desc="RN quantizing"):
            stats = quantize_linear_nearest(
                module,
                group_size=args.group_size,
                chunk_rows=args.chunk_rows,
                weight_loader=weight_loader,
                weight_name=f"{name}.weight",
            )
            layer_stats[name] = stats
            if stats.get("loaded_from_safetensors"):
                weight_loader.clear_cache()

    total_numel = sum(stats["numel"] for stats in layer_stats.values())
    mean_mse = sum(stats["mse"] * stats["numel"] for stats in layer_stats.values()) / max(total_numel, 1)
    mean_mae = sum(stats["mae"] * stats["numel"] for stats in layer_stats.values()) / max(total_numel, 1)
    max_error = max((stats["max_error"] for stats in layer_stats.values()), default=0.0)
    meta_fallback_layers = sum(1 for stats in layer_stats.values() if stats.get("loaded_from_safetensors"))
    remaining_meta_parameters = materialize_remaining_meta_parameters(model, weight_loader)
    print(f"\nMeta linear layers loaded from safetensors: {meta_fallback_layers}")
    print(f"Remaining meta parameters materialized:    {remaining_meta_parameters}")

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
        "meta_fallback_layers": meta_fallback_layers,
        "remaining_meta_parameters_materialized": remaining_meta_parameters,
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
