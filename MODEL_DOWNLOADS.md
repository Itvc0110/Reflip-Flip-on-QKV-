# Model And Data Download Guide

This repo can run in two styles:

1. Local model directory, such as `./models/Llama-3-8B`.
2. Hugging Face model id, such as `openbmb/MiniCPM-2B-sft-bf16`.

The standalone QKV runbook currently expects local `./models/Llama-3-8B` because `xspot.py`, `export_qkv_gqa.py`, `quantize_qkv.py`, and `fast_quantize_qkv.py` are aligned around that local path and the hardcoded `./xspot_layer5_group2` export.

## Install Download Tooling

The `huggingface-cli` command comes from `huggingface_hub`:

```bash
python -m pip install huggingface_hub
```

For gated models, log in first:

```bash
huggingface-cli login
```

## First Standalone QKV Model

Recommended first local path:

```text
./models/Llama-3-8B
```

If you have access to a Llama 3 8B repository on Hugging Face, download it into that path:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir ./models/Llama-3-8B --local-dir-use-symlinks False
```

If you use a different Llama 3 or Llama 3.1 8B source, keep the local destination the same:

```bash
huggingface-cli download <your-llama-3-8b-repo-id> --local-dir ./models/Llama-3-8B --local-dir-use-symlinks False
```

After download:

```bash
python tools/preflight_qkv.py
python tools/inspect_model_config.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2
python tools/preflight_full_eval.py
```

If the model path is present, the standalone preflight will move you to the `xspot.py` export command. The full-model preflight will move you to the first AWQ baseline command.

For later full-model commands, see `RUN_FULL_MODEL_EVAL.md`.

## AWQ XL Default Model

Several full-model AWQ scripts default to:

```text
./models/Mistral-7B-v0.3
```

Download command:

```bash
huggingface-cli download mistralai/Mistral-7B-v0.3 --local-dir ./models/Mistral-7B-v0.3 --local-dir-use-symlinks False
```

Scripts that default to this path:

- `awq_stand_xl.py`
- `awq_dh_xl.py`
- `awq_js_xl.py`

You can also pass `--model-path ./models/Llama-3-8B` to these scripts when you want to use the same model as the standalone QKV path.

## MiniCPM Scripts

Older MiniCPM scripts load this Hugging Face id directly:

```text
openbmb/MiniCPM-2B-sft-bf16
```

To predownload it locally:

```bash
huggingface-cli download openbmb/MiniCPM-2B-sft-bf16 --local-dir ./models/MiniCPM-2B-sft-bf16 --local-dir-use-symlinks False
```

Scripts that reference MiniCPM directly include:

- `gw_awq_asym_l2.py`
- `gw_awq_asym_l2_stats.py`
- `export_data.py`
- `analyze_saliency_tail.py`
- `calibration_utils.py` test block

Some of these scripts have the model id hardcoded, so use them as-is first unless you decide to refactor model-path configurability later.

## Calibration And Evaluation Data

These are downloaded automatically by `datasets` when the scripts run:

- C4 calibration shard via `calibration_utils.py`
- WikiText-2 train/test
- C4 validation
- AG News test
- lm-eval task datasets later

Local cache directories used by current scripts:

```text
./calibration_cache
./dataset_cache
```

No manual dataset download is required for the first standalone QKV run. The model is the missing manual step.

## Large Generated Outputs

Do not commit these unless intentionally creating a tiny fixture:

```text
./models/
./xspot_layer*/
./quantized_models/
./calibration_cache/
./dataset_cache/
./results/lm_eval/
attention_quantization_analysis.png
sorted_error_comparison.png
quantization_results.npz
```
