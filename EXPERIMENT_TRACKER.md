# Flip/ReFlip Experiment Tracker

This tracker connects the current repo files to the larger experiment goal: run Flip/ReFlip and related kneedle/James-Stein methods on standalone QKV first, then later compare full-model quantization baselines such as RN, AWQ, dynamic heuristic AWQ, and AWQ+GQA ReFlip.

`Smart-Flip Quantization Results.xlsx` is reference history. New runs should be recorded in `results/experiment_ledger.jsonl`.

Workbook metric notes are summarized in `SMART_FLIP_REFERENCE.md`.
Model download commands are summarized in `MODEL_DOWNLOADS.md`.
Full-model commands are summarized in `RUN_FULL_MODEL_EVAL.md`.
Machine-readable workflow metadata is in `configs/experiment_manifest.json`.

## Current Status

| Area | Status | Evidence / next artifact |
| --- | --- | --- |
| Repo inspection | Done | Existing scripts mapped in this tracker and `RUN_QKV_STANDALONE.md` |
| No-model sanity check | Done | `heuristic_verification.py` run recorded in `results/experiment_ledger.jsonl` |
| Standalone QKV export | To do | Generate `./xspot_layer5_group2/*.npy` with `xspot.py` |
| Standalone QKV Flip/ReFlip metrics | To do | Run `fast_quantize_qkv.py` after QKV export exists |
| Full-model RN / AWQ comparison | Later | Use full-model scripts after standalone path is verified |
| lm-eval comparison | Later | Requires quantized model outputs and `lm-eval` install |

## Files And Roles

| File | Role | Run now? |
| --- | --- | --- |
| `utils_qkv.py` | Shared Kneedle, dynamic outlier threshold, nearest INT4 quantization, quantization error | Library only |
| `heuristic_verification.py` | Synthetic no-model check for greedy quantization error reduction using `data10.csv` | Yes |
| `tools/experiment_status.py` | Prints a compact status snapshot and runs both preflights | Yes |
| `tools/validate_experiment_manifest.py` | Validates manifest references against current repo files | Yes |
| `tools/print_experiment_commands.py` | Prints download, standalone, and full-model commands from the manifest | Yes |
| `tools/preflight_qkv.py` | Checks dependencies, expected model path, and generated QKV artifacts | Yes |
| `tools/preflight_full_eval.py` | Checks later full-model/lm-eval packages, base model, and quantized outputs | Later |
| `tools/inspect_model_config.py` | Reads local `config.json` to validate layer/group/head compatibility before loading weights | After model download |
| `tools/inspect_smart_flip_workbook.py` | Prints sheet previews from the reference workbook | Optional |
| `tools/summarize_qkv_results.py` | Summarizes `quantization_results.npz` and can append metrics to the ledger | After QKV run |
| `tools/summarize_lm_eval.py` | Summarizes lm-eval JSON outputs and can append workbook-style task scores to the ledger | After lm-eval |
| `RUN_FULL_MODEL_EVAL.md` | Step-by-step full-model quantization and lm-eval runbook | Later |
| `configs/experiment_manifest.json` | Machine-readable method, command, model, and artifact manifest | Reference |
| `xspot.py` | Exports James-Stein means and one GQA Q/K/V group from a model | Yes, once model is available |
| `fast_quantize_qkv.py` | Fast standalone nearest vs Flip vs ReFlip comparison | Yes, after `xspot.py` |
| `quantize_qkv.py` | Slower standalone nearest vs Flip vs ReFlip comparison | Optional |
| `test_quantize_qkv.py` | Older standalone nearest vs Flip comparison | Optional / legacy |
| `visualize_xspot.py` | Plots exported James-Stein and Q/K/V group data | Optional |
| `rtn_xl.py` | Full-model nearest/RN INT4 groupwise baseline, saved as dequantized HF model | Later |
| `awq_stand_xl.py` | Full-model standard AWQ baseline | Later |
| `awq_dh_xl.py` | Full-model dynamic heuristic AWQ with Kneedle outliers and Flip | Later |
| `awq_js_xl.py` | Full-model James-Stein heuristic AWQ | Later |
| `awq_gqa_xl.py` | Full-model AWQ + GQA ReFlip refinement | Later |
| `compare_awq_heuristic.py` | Perplexity comparison for two saved model directories | Later |
| `compare_awq_slicing.py` | Perplexity comparison for heuristic vs standard model directories | Later |
| `export_qkv_gqa.py` | Exports broader GQA activations/weights, not the exact files used by `quantize_qkv.py` | Optional |
| `analyze_saliency_tail.py` | Saliency/tail/knee analysis; needs model/datasets and `kneed` | Optional |
| `requirements-qkv.txt` | Install list for standalone QKV scripts | Setup |
| `requirements-full-eval.txt` | Extra install list for full-model evaluation and lm-eval | Later setup |
| `SMART_FLIP_REFERENCE.md` | Notes on workbook sheets, metrics, and labels | Reference |
| `MODEL_DOWNLOADS.md` | Exact model/download paths and commands for referenced scripts | Setup |

## Download / Generated File Checklist

Install dependencies for the standalone path:

```bash
python -m pip install -r requirements-qkv.txt
```

Install dependencies for later full-model evaluation and optional saliency analysis:

```bash
python -m pip install -r requirements-full-eval.txt
```

Model files needed for the first standalone QKV run:

```text
./models/Llama-3-8B/
```

Download/setup details are in `MODEL_DOWNLOADS.md`.

Generated by `xspot.py`:

```text
./xspot_layer5_group2/X_activations.npy
./xspot_layer5_group2/naive_means.npy
./xspot_layer5_group2/js_means.npy
./xspot_layer5_group2/grand_mean.npy
./xspot_layer5_group2/shrinkage_factor.npy
./xspot_layer5_group2/Wq_group2.npy
./xspot_layer5_group2/Wk_group2.npy
./xspot_layer5_group2/Wv_group2.npy
./xspot_layer5_group2/metadata.json
./xspot_layer5_group2/README.txt
```

Generated by `fast_quantize_qkv.py`:

```text
attention_quantization_analysis.png
sorted_error_comparison.png
quantization_results.npz
```

Generated later by full-model quantization:

```text
./quantized_models/<method_name>/
```

Calibration data is downloaded automatically by the scripts into `./calibration_cache`.

## Immediate Run Queue

1. Check overall status:

```bash
python tools/experiment_status.py
python tools/print_experiment_commands.py --section standalone
```

2. Confirm standalone readiness:

```bash
python tools/preflight_qkv.py
python tools/inspect_model_config.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2
```

3. Generate the standalone QKV artifacts:

```bash
python xspot.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2 --n-samples 128 --seqlen 512 --output-dir ./xspot_layer5_group2
```

4. Run standalone QKV metrics:

```bash
python fast_quantize_qkv.py --critical-dim-pct 0.1 --knee-tolerance 0.0 --group-size 128 --max-flip-pct 0.1 --correction-scale 1.0
```

5. Record the printed summary metrics in `results/experiment_ledger.jsonl`.

```bash
python tools/summarize_qkv_results.py --append-ledger
```

## Metrics To Record

Standalone QKV:

- Mean absolute attention-score error for `Nearest`, `Heuristic`, and `ReFlip`.
- Mean relative attention-score error for `Nearest`, `Heuristic`, and `ReFlip`.
- Mean improvement for `Nearest -> Heuristic`, `Nearest -> ReFlip`, and `Heuristic -> ReFlip`.
- Total Flip count, Flip rate, outlier percentage.
- ReFlip moderate dimensions, ReFlip integer flips, ReFlip flip rate.

Full-model later:

- WikiText-2 perplexity.
- C4 perplexity.
- lm-eval task scores matching the reference workbook when possible: `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `lambada`, `openbookqa`, `piqa`, `rte`, `winogrande`.
- Average task score using the same task subset for every compared run.

## Later Full-Model Queue

Check later-stage readiness:

```bash
python tools/preflight_full_eval.py
```

Standard AWQ baseline:

```bash
python awq_stand_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_standard --n-calib 128 --layer-batch-size 16
```

Dynamic heuristic AWQ:

```bash
python awq_dh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_dh --n-calib 128 --knee-tolerance 0.0 --max-flip-percent 0.01 --layer-batch-size 16
```

James-Stein heuristic AWQ:

```bash
python awq_js_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_js --n-calib 128 --knee-tolerance 0.0 --max-flip-percent 0.05 --layer-batch-size 16
```

AWQ + GQA ReFlip:

```bash
python awq_gqa_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/llama3_awq_gqa --n-calib 128 --apply-gqa-reflip --gqa-critical-dim-pct 0.15 --gqa-max-flip-pct 0.05
```

Perplexity comparison after two model outputs exist:

```bash
python compare_awq_slicing.py --heuristic-path ./quantized_models/llama3_awq_dh --standard-path ./quantized_models/llama3_awq_standard --n-samples 500
```

lm-eval examples after a quantized model exists:

```bash
lm_eval --model hf --model_args pretrained=./quantized_models/llama3_awq_standard,trust_remote_code=True --tasks arc_challenge,arc_easy,boolq,piqa,rte --device cuda --batch_size auto --output_path ./results/lm_eval/llama3_awq_standard
```

```bash
lm_eval --model hf --model_args pretrained=./quantized_models/llama3_awq_gqa,trust_remote_code=True --tasks arc_challenge,arc_easy,boolq,piqa,rte --device cuda --batch_size auto --output_path ./results/lm_eval/llama3_awq_gqa
```

Summarize and record lm-eval JSON outputs:

```bash
python tools/summarize_lm_eval.py --path ./results/lm_eval --append-ledger
```

## Notes

- Existing docs mention some scripts that are not present, including `awq_sh.py`, `awq_op_ref.py`, `final_cross_validation.py`, and `quantize_autoawq_library.py`. Prefer commands that correspond to files present in this repo.
- The standalone scripts currently use hardcoded QKV directories. For the first pass, match those paths instead of editing code.
- Generated model outputs, calibration caches, and `.npy` QKV exports can be large. Do not commit them unless a tiny fixture is intentionally created.
