"""
RTN-GQA: Plain RTN Quantization with GQA-Specific Flip/ReFlip Refinement

Same pipeline as awq_gqa_xl.py, with the AWQ activation-based column scaling
step forced to identity (scale=1, alpha=0). That degrades the base quantizer
from AWQ to plain group-wise round-to-nearest (RTN): scale=(w_max-w_min)/15,
zero-point=round(-w_min/scale), matching utils_qkv.py's quantize_weight_groupwise_int4.
The Flip (--use-heuristic) and GQA-aware ReFlip (--apply-gqa-reflip) stages
run completely unmodified on top, since both only ever consume "best_scales"
as an opaque per-channel tensor.

Pipeline:
1. Plain RTN quantization for all linear layers (AWQGQAQuantizer with identity scale)
2. Detect Group-Query Attention layers (Q_proj, K_proj, V_proj)
3. Apply Flip (global greedy rounding correction) during step 1 if --use-heuristic
4. Apply ReFlip refinement to GQA layers if --apply-gqa-reflip

This produces the "RTN", "RTN + Flip", and "RTN + ReFlip" full-model
checkpoints reported in the thesis (RTN + ReFlip is structurally
RTN -> Flip -> ReFlip, matching the standalone case study's progression).

Usage:
    # RTN + Flip + ReFlip (the thesis's "RTN + ReFlip" full-model row)
    python rtn_gqa_xl.py \
        --model-path ./models/Llama-3-8B \
        --output-dir ./quantized_models/llama3_rtn_reflip \
        --n-calib 128 \
        --apply-gqa-reflip

    # RTN + Flip only (no ReFlip)
    python rtn_gqa_xl.py \
        --model-path ./models/Llama-3-8B \
        --output-dir ./quantized_models/llama3_rtn_flip \
        --n-calib 128

Plain RTN with no correction at all is cheaper to produce via rtn_xl.py directly
(no calibration pass needed); this script's --no-heuristic mode reproduces the
same result but is not the recommended path for that baseline.
"""

import torch
import hf_runtime  # noqa: F401 - sets Transformers backend env vars before import.
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
import os
import random

# Reuse the GQA-ReFlip pipeline unmodified; only the AWQ scale search is overridden below.
from awq_gqa_xl import AWQGQAQuantizer


class RTNGQAQuantizer(AWQGQAQuantizer):
    """AWQGQAQuantizer with the AWQ activation-scaling step forced to identity.

    Every consumer of `best_scales` (quantize_layer, refine_attention_group)
    treats it as an opaque per-channel tensor with no invariant beyond matching
    `in_features`, so returning an all-ones vector degrades the base quantizer
    to plain group-wise RTN with zero other code changes. Flip and GQA-ReFlip
    run unmodified on top via plain inheritance.
    """

    def _weight_device(self, module):
        """Device the module's weight actually lives on (multi-GPU device_map support).

        With device_map="auto" on multi-GPU boxes (e.g. Kaggle T4 x2) the model is
        sharded across cuda:0/cuda:1. The parent quantizer creates helper tensors on
        self.device (cuda:0), which raises "Expected all tensors to be on the same
        device" for every layer placed on cuda:1 — silently leaving those layers
        UNQUANTIZED. All overrides below therefore follow the weight's own device.
        """
        w = module.weight
        if getattr(w, "is_meta", False):
            return self.device
        return w.device

    def search_best_scale(self, name, module):
        in_features = module.weight.shape[1]
        scales = torch.ones(in_features, device=self._weight_device(module),
                            dtype=module.weight.dtype)
        return scales, 0.0, 0.0

    def search_best_scale_lmhead_half(self, name, module, out_start, out_end, debug=False):
        in_features = module.weight.shape[1]
        scales = torch.ones(in_features, device=self._weight_device(module),
                            dtype=module.weight.dtype)
        return scales, 0.0, 0.0

    def quantize_layer(self, name, module):
        # Temporarily point self.device at this layer's device so every
        # ".to(self.device)" inside the inherited quantization path lands on the
        # same GPU as the weight. refine_attention_group already uses the module's
        # own device and needs no hop.
        prev_device = self.device
        try:
            self.device = self._weight_device(module)
            super().quantize_layer(name, module)
        finally:
            self.device = prev_device

    def quantize_lmhead_half_by_half(self, name, module, debug=False, num_chunks=4):
        prev_device = self.device
        try:
            self.device = self._weight_device(module)
            super().quantize_lmhead_half_by_half(name, module, debug=debug,
                                                 num_chunks=num_chunks)
        finally:
            self.device = prev_device


def main():
    parser = argparse.ArgumentParser(description='RTN-GQA: Plain RTN Quantization with GQA ReFlip Refinement')

    # All arguments from awq_gqa_xl.py (n-grid kept for CLI compatibility; unused since scale search is skipped)
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--use-heuristic", action="store_true", default=True,
                        help="Enable Flip (global greedy rounding correction)")
    parser.add_argument("--no-heuristic", dest="use_heuristic", action="store_false",
                        help="Disable Flip; produces plain RTN (prefer rtn_xl.py for this)")
    parser.add_argument("--use-james-stein", action="store_true", default=True)
    parser.add_argument("--no-james-stein", dest="use_james_stein", action="store_false")
    parser.add_argument("--knee-tolerance", type=float, default=0.000)
    parser.add_argument("--max-flip-percent", type=float, default=0.05)
    parser.add_argument("--max-tokens-per-sample", type=int, default=2048)
    parser.add_argument("--layer-batch-size", type=int, default=16)
    parser.add_argument("--lmhead-chunks", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="./quantized_models/llama3_rtn_gqa")
    parser.add_argument("--model-path", type=str, default="./models/Llama-3-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-dataset", type=str, default="c4",
                        choices=["c4", "wikitext2", "wikitext2-simple"])
    parser.add_argument("--cache-dir", type=str, default="./calibration_cache")

    # GQA ReFlip options
    parser.add_argument('--apply-gqa-reflip', action='store_true',
                        help='Apply ReFlip refinement to GQA layers')
    parser.add_argument('--gqa-critical-dim-pct', type=float, default=0.15)
    parser.add_argument('--gqa-max-flip-pct', type=float, default=0.05)

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("\n" + "=" * 80)
    print("RTN-GQA: Plain RTN Quantization with GQA ReFlip Refinement")
    print("=" * 80)
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Calibration samples: {args.n_calib}")
    print(f"  Flip: {'Enabled' if args.use_heuristic else 'Disabled (plain RTN)'}")
    print(f"  GQA ReFlip: {'Enabled' if args.apply_gqa_reflip else 'Disabled'}")
    if args.apply_gqa_reflip:
        print(f"    - Critical dim %: {args.gqa_critical_dim_pct}")
        print(f"    - Max flip %: {args.gqa_max_flip_pct}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"\nLoading model and tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  -> Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    # Create quantizer with GQA support (AWQ scale search forced to identity -> plain RTN base)
    quantizer = RTNGQAQuantizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bits=args.bits,
        n_grid=args.n_grid,
        group_size=args.group_size,
        use_heuristic=args.use_heuristic,
        knee_tolerance=args.knee_tolerance,
        max_tokens_per_sample=args.max_tokens_per_sample,
        layer_batch_size=args.layer_batch_size,
        lmhead_chunks=args.lmhead_chunks,
        max_flip_percent=args.max_flip_percent,
        use_james_stein=args.use_james_stein,
        apply_gqa_reflip=args.apply_gqa_reflip,
        gqa_critical_dim_pct=args.gqa_critical_dim_pct,
        gqa_max_flip_pct=args.gqa_max_flip_pct
    )

    # Load calibration data
    from calibration_utils import get_c4_calibration_data, get_wikitext2_calibration_data

    print(f"\nLoading calibration dataset: {args.calib_dataset}")
    if args.calib_dataset == "c4":
        calib_texts = get_c4_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)
    elif args.calib_dataset == "wikitext2-simple":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        calib_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100][:args.n_calib]
    else:
        calib_texts = get_wikitext2_calibration_data(quantizer.tokenizer, n_samples=args.n_calib, seqlen=2048, seed=args.seed, cache_dir=args.cache_dir)

    # Quantize model
    quantizer.quantize_model_sequential(calib_texts, n_samples=args.n_calib)

    # Save quantized model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()
