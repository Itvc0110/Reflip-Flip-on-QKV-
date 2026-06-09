"""Inspect a local Hugging Face model config for QKV/GQA compatibility.

This helper reads only `config.json`. It does not load model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local model config for GQA/QKV runs")
    parser.add_argument("--model-path", default="./models/Llama-3-8B")
    parser.add_argument("--layer-id", type=int, default=5)
    parser.add_argument("--group-id", type=int, default=2)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model directory not found: {model_path}")
        print("Download/place the model first. See MODEL_DOWNLOADS.md.")
        return 1

    try:
        config = read_config(model_path)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    model_type = config.get("model_type", "unknown")
    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers") or config.get("n_layer")
    num_heads = config.get("num_attention_heads") or config.get("n_head")
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim")
    if head_dim is None and hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    print("=" * 80)
    print("Model Config Inspection")
    print("=" * 80)
    print(f"Model path:          {model_path}")
    print(f"model_type:          {model_type}")
    print(f"hidden_size:         {hidden_size}")
    print(f"num_hidden_layers:   {num_layers}")
    print(f"num_attention_heads: {num_heads}")
    print(f"num_key_value_heads: {num_kv_heads}")
    print(f"head_dim:            {head_dim}")

    if not all(isinstance(v, int) and v > 0 for v in [hidden_size, num_layers, num_heads, num_kv_heads, head_dim]):
        print("\nCould not infer all required attention dimensions from config.json.")
        return 1

    queries_per_kv = num_heads // num_kv_heads
    is_gqa = num_kv_heads < num_heads
    print(f"queries_per_kv:      {queries_per_kv}")
    print(f"GQA/MQA detected:    {'yes' if is_gqa else 'no'}")

    problems = []
    if hidden_size % num_heads != 0:
        problems.append("hidden_size is not divisible by num_attention_heads")
    if num_heads % num_kv_heads != 0:
        problems.append("num_attention_heads is not divisible by num_key_value_heads")
    if args.layer_id < 0 or args.layer_id >= num_layers:
        problems.append(f"layer-id {args.layer_id} is outside [0, {num_layers - 1}]")
    if args.group_id < 0 or args.group_id >= num_kv_heads:
        problems.append(f"group-id {args.group_id} is outside [0, {num_kv_heads - 1}]")

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("\nConfiguration looks compatible with the standalone xspot.py group export.")
    print("\nSuggested export command:")
    print(
        f"python xspot.py --model-path {args.model_path} --layer-id {args.layer_id} "
        f"--group-id {args.group_id} --n-samples 128 --seqlen 512 "
        f"--output-dir ./xspot_layer{args.layer_id}_group{args.group_id}"
    )
    if args.layer_id == 5 and args.group_id == 2:
        print("\nThis matches quantize_qkv.py's current hardcoded ./xspot_layer5_group2 path.")
    else:
        print("\nNote: quantize_qkv.py is currently hardcoded to ./xspot_layer5_group2.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
