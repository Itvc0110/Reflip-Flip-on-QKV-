# Full-Model Quantization And lm-eval Runbook

This runbook is for the later phase after the standalone QKV path works. It explains how to run the full-model scripts that exist in this repo and how to compare them using perplexity and lm-eval style metrics.

Use `Smart-Flip Quantization Results.xlsx` only as a reference template. New results should go into `results/experiment_ledger.jsonl`.

## 1. Readiness Check

Run:

```bash
python tools/preflight_full_eval.py
```

If packages are missing:

```bash
python -m pip install -r requirements-full-eval.txt
```

If the model is missing, see `MODEL_DOWNLOADS.md`. The commands below assume:

```text
./models/Llama-3-8B
```

## 2. Methods In This Repo

| Method | Script | Main idea |
| --- | --- | --- |
| RN / RTN | `rtn_xl.py` | Full-model nearest INT4 groupwise asymmetric quantization, saved as dequantized weights |
| Standard AWQ | `awq_stand_xl.py` | L2 salience AWQ + nearest INT4 groupwise asymmetric quantization |
| Dynamic heuristic AWQ | `awq_dh_xl.py` | AWQ + Kneedle dynamic outlier masking + Flip |
| James-Stein heuristic AWQ | `awq_js_xl.py` | AWQ + James-Stein activation means + heuristic Flip |
| AWQ + GQA ReFlip | `awq_gqa_xl.py` | James-Stein heuristic AWQ plus attention-group ReFlip refinement |

`rtn_xl.py` is an additive baseline script. It uses `utils_qkv.quantize_weight_groupwise_int4` and stores dequantized FP16/BF16 weights for research comparison, not packed INT4 kernels.

## 3. Suggested First Full-Model Sequence

Start with a smaller calibration count if you are testing plumbing. Use the same `--model-path` for every method so comparisons are fair.

### RN / RTN

```bash
python rtn_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_rtn_w4g128 --group-size 128
```

### Standard AWQ

```bash
python awq_stand_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_standard --n-calib 128 --layer-batch-size 16
```

### Dynamic Heuristic AWQ

```bash
python awq_dh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_dh --n-calib 128 --knee-tolerance 0.0 --max-flip-percent 0.01 --layer-batch-size 16
```

### James-Stein Heuristic AWQ

```bash
python awq_js_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_js --n-calib 128 --knee-tolerance 0.0 --max-flip-percent 0.05 --layer-batch-size 16
```

### AWQ + GQA ReFlip

```bash
python awq_gqa_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_gqa --n-calib 128 --apply-gqa-reflip --gqa-critical-dim-pct 0.15 --gqa-max-flip-pct 0.05
```

## 4. Perplexity Comparison

After two quantized model directories exist, run:

```bash
python compare_awq_slicing.py --heuristic-path ./quantized_models/llama3_awq_dh --standard-path ./quantized_models/llama3_awq_standard --n-samples 500
```

For a larger run:

```bash
python compare_awq_slicing.py --heuristic-path ./quantized_models/llama3_awq_gqa --standard-path ./quantized_models/llama3_awq_standard --n-samples 2000
```

Record WikiText-2 and C4 perplexity in `results/experiment_ledger.jsonl`.

## 5. lm-eval Comparison

Start with the task subset used repeatedly in the workbook:

```bash
lm_eval --model hf --model_args pretrained=./quantized_models/llama3_awq_standard,trust_remote_code=True --tasks arc_challenge,arc_easy,boolq,piqa,rte --device cuda --batch_size auto --output_path ./results/lm_eval/llama3_awq_standard
```

```bash
lm_eval --model hf --model_args pretrained=./quantized_models/llama3_awq_gqa,trust_remote_code=True --tasks arc_challenge,arc_easy,boolq,piqa,rte --device cuda --batch_size auto --output_path ./results/lm_eval/llama3_awq_gqa
```

Then summarize:

```bash
python tools/summarize_lm_eval.py --path ./results/lm_eval --append-ledger
```

For a workbook-like expanded task set, use:

```bash
lm_eval --model hf --model_args pretrained=./quantized_models/llama3_awq_gqa,trust_remote_code=True --tasks arc_challenge,arc_easy,boolq,hellaswag,lambada_openai,openbookqa,piqa,rte,winogrande --device cuda --batch_size auto --output_path ./results/lm_eval/llama3_awq_gqa_full
```

## 6. Metrics To Compare

Lower is better:

- WikiText-2 perplexity
- C4 perplexity

Higher is better:

- `arc-c` / `arc_challenge`
- `arc-e` / `arc_easy`
- `boolq`
- `hellaswag`
- `lambada`
- `openbookqa`
- `piqa`
- `rte`
- `winogrande`
- average over an identical task subset

Use `SMART_FLIP_REFERENCE.md` for workbook naming notes.

## 7. Recording Policy

For every full-model run, record:

- script and method label
- model path
- output directory
- bit width and group size
- calibration dataset, sample count, and sequence length
- knee tolerance and flip percent where applicable
- GQA ReFlip settings where applicable
- perplexity results
- lm-eval task list and scores

Use:

```bash
python tools/summarize_lm_eval.py --path ./results/lm_eval --append-ledger
```

For manual rows, append one JSON object per line to:

```text
results/experiment_ledger.jsonl
```

## 8. Current Known Blockers

On the current machine state, `tools/preflight_full_eval.py` reports:

- missing `./models/Llama-3-8B`
- missing `kneed`
- missing `lm-eval`
- missing quantized model output directories

That is expected before the model download and later-stage dependency install.
