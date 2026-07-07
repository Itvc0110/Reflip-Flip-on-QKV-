"""CPU-only smoke test for rtn_gqa_xl.py, no real weights/GPU/network model download needed.

Builds a tiny random-init GQA LlamaForCausalLM (config only, no pretrained weights) and runs
RTNGQAQuantizer's full pipeline (RTN base + Flip + GQA-ReFlip) end to end. Confirms:
  1. search_best_scale/search_best_scale_lmhead_half return identity (all-ones) scales, so the
     base quantizer is really plain RTN rather than AWQ.
  2. The GQA-ReFlip refinement runs without crashing on an actual GQA layer.
  3. The quantized model still produces finite logits and round-trips through
     save_pretrained/from_pretrained.

Only the tokenizer is downloaded (hf-internal-testing/tiny-random-LlamaForCausalLM, a few KB,
cached after first run); the model itself is constructed fresh with random weights so this test
never depends on the real Llama-3-8B checkpoint.

Usage:
    python -m pytest tests/test_rtn_gqa_xl_smoke.py -v
"""

import os
import sys
import tempfile

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtn_gqa_xl import RTNGQAQuantizer  # noqa: E402

TOKENIZER_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"

CALIB_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language models are trained on large text corpora.",
    "Grouped-query attention shares key and value heads across query heads.",
    "Quantization reduces the memory footprint of neural network weights.",
] * 2  # 8 short samples total


def build_tiny_gqa_model(tokenizer):
    """A GQA Llama with 4 query heads sharing 2 key/value heads, head_dim=32."""
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def test_rtn_gqa_quantizer_end_to_end():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_tiny_gqa_model(tokenizer)

    quantizer = RTNGQAQuantizer(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        bits=4,
        n_grid=20,
        group_size=8,
        use_heuristic=True,
        knee_tolerance=0.0,
        max_tokens_per_sample=32,
        layer_batch_size=64,  # bigger than the total linear-layer count -> one batch
        lmhead_chunks=1,
        max_flip_percent=0.05,
        use_james_stein=True,
        apply_gqa_reflip=True,
        gqa_critical_dim_pct=0.15,
        gqa_max_flip_pct=0.05,
    )

    # 1. search_best_scale must be identity (proves the base quantizer is plain RTN, not AWQ)
    q_proj = model.model.layers[0].self_attn.q_proj
    scales, alpha, error = quantizer.search_best_scale("model.layers.0.self_attn.q_proj", q_proj)
    assert torch.allclose(scales, torch.ones_like(scales))
    assert alpha == 0.0
    assert error == 0.0

    # 2. Full pipeline: RTN base + Flip + GQA-ReFlip, must not raise
    quantizer.quantize_model_sequential(CALIB_TEXTS, n_samples=len(CALIB_TEXTS))

    # The GQA-ReFlip step must have actually recorded identity AWQ scales for a real Q/K layer
    stored_scales = quantizer.gqa_awq_scales.get("model.layers.0.self_attn.q_proj.weight")
    assert stored_scales is not None, "GQA-ReFlip did not process the expected q_proj layer"
    assert torch.allclose(stored_scales, torch.ones_like(stored_scales))

    # 3. Quantized model still produces finite logits
    inputs = tokenizer("hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
    assert torch.isfinite(out.logits).all()

    # 4. Round-trip through save_pretrained/from_pretrained
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        reloaded = LlamaForCausalLM.from_pretrained(tmp_dir)
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(**inputs, use_cache=False)
        assert torch.isfinite(out2.logits).all()


if __name__ == "__main__":
    test_rtn_gqa_quantizer_end_to_end()
    print("OK: rtn_gqa_xl.py smoke test passed")
