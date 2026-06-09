# Standalone QKV Flip/ReFlip Runbook

This runbook explains how to run the existing standalone QKV scripts and how to read what they print. It treats `Smart-Flip Quantization Results.xlsx` as reference history only. The first goal is to reproduce the current script outputs without editing quantization code.

## 1. What This Path Runs

The standalone QKV path is a local attention-group experiment, not full-model quantization.

- `xspot.py` loads a model, captures Q-projection input activations, computes James-Stein channel means, and exports one Q/K/V GQA group.
- `fast_quantize_qkv.py` reads the exported files and compares nearest rounding, heuristic Flip, and ReFlip.
- `quantize_qkv.py` is the slower original implementation of the same comparison.
- `visualize_xspot.py` optionally plots the exported James-Stein and Q/K/V data.
- `heuristic_verification.py` is a small no-model sanity check using `data10.csv`.

Important current limitation: `fast_quantize_qkv.py` and `quantize_qkv.py` call the same `quantize_qkv.py` main function, which is hardcoded to read `./xspot_layer5_group2`. For the first run, export exactly that directory.

## 2. Files Needed

Install dependencies:

```bash
python -m pip install -r requirements-qkv.txt
```

Provide or download a local model:

```text
./models/Llama-3-8B
```

See `MODEL_DOWNLOADS.md` for exact `huggingface-cli` commands.

The QKV scripts need these generated files:

```text
./xspot_layer5_group2/js_means.npy
./xspot_layer5_group2/Wq_group2.npy
./xspot_layer5_group2/Wk_group2.npy
./xspot_layer5_group2/Wv_group2.npy
./xspot_layer5_group2/metadata.json
```

These files are produced by `xspot.py`; they are not committed in the repo.

## 3. Step-By-Step Commands

Check what is installed and what is still missing:

```bash
python tools/experiment_status.py
python tools/preflight_qkv.py
```

After the model is downloaded, check the model config before loading weights:

```bash
python tools/inspect_model_config.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2
```

Run a no-model sanity check first:

```bash
python heuristic_verification.py
```

Expected current output:

```text
Data Loaded. D=2304. Group Size=128

--- RESULTS (Absolute Dot Product Error) ---
1. Non-Group Global Proposed:  0.004118
2. Group-Wise Nearest (Base):  0.001875
3. Group-Wise Proposed (New):  0.001396

--- COMPARISON ---
New Method vs GW Nearest: 1.34x improvement
New Method vs Non-Group:  2.95x improvement
```

Export QKV data for the hardcoded standalone path:

```bash
python xspot.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2 --n-samples 128 --seqlen 512 --output-dir ./xspot_layer5_group2
```

For a quick smoke run:

```bash
python xspot.py --model-path ./models/Llama-3-8B --layer-id 5 --group-id 2 --n-samples 8 --seqlen 128 --output-dir ./xspot_layer5_group2
```

Optional visualization:

```bash
python visualize_xspot.py --data-dir ./xspot_layer5_group2
```

Run the standalone QKV metrics:

```bash
python fast_quantize_qkv.py --critical-dim-pct 0.1 --knee-tolerance 0.0 --group-size 128 --max-flip-pct 0.1 --correction-scale 1.0
```

Summarize and optionally record the generated `quantization_results.npz`:

```bash
python tools/summarize_qkv_results.py
python tools/summarize_qkv_results.py --append-ledger
```

Use the slower original only if comparing implementation behavior:

```bash
python quantize_qkv.py --critical-dim-pct 0.1 --knee-tolerance 0.0 --group-size 128 --max-flip-pct 0.1 --correction-scale 1.0
```

## 4. What The Printed Metrics Mean

- `James-Stein Statistics`: how activation channel means are shrunk toward the grand mean.
- `Weight quantization errors`: MAE, max absolute error, and relative error between original weights and dequantized weights.
- `Original score`: attention score `(X @ Wq.T) dot (X @ Wk.T)` for each query head.
- `Nearest`: standard round-to-nearest INT4 groupwise asymmetric quantization.
- `Heuristic` / `Flip`: greedy integer flip correction guided by activation means and Kneedle outlier masking.
- `ReFlip`: second-stage correction targeting Q/K attention-score error on moderate dimensions.
- `Absolute error`: quantized attention score minus original score.
- `Relative error`: score error divided by original score magnitude.
- `Nearest -> Heuristic`: percent reduction in absolute attention-score error from nearest to Flip.
- `Nearest -> ReFlip`: percent reduction in absolute attention-score error from nearest to ReFlip.
- `Heuristic -> ReFlip`: percent reduction in absolute attention-score error from Flip to ReFlip.
- `total flips` / `flip rate`: how many INT4 weights were changed by plus or minus one.
- `outlier percent`: channels protected by Kneedle-based dynamic outlier detection.
- `moderate dims`: Q dimensions selected by Kneedle for ReFlip correction.

## 5. Outputs To Check

After `fast_quantize_qkv.py`, expect:

```text
attention_quantization_analysis.png
sorted_error_comparison.png
quantization_results.npz
```

After `visualize_xspot.py`, expect visualizations under:

```text
./xspot_layer5_group2/visualizations/
```

## 6. Tracking Runs

Use `results/experiment_ledger.jsonl` for one JSON object per run. Suggested fields:

```json
{"status":"todo","script":"fast_quantize_qkv.py","model_path":"./models/Llama-3-8B","data_dir":"./xspot_layer5_group2","layer_id":5,"group_id":2,"n_samples":128,"seqlen":512,"critical_dim_pct":0.1,"knee_tolerance":0.0,"group_size":128,"max_flip_pct":0.1,"correction_scale":1.0,"notes":"Fill metrics after run"}
```

Keep generated model/data artifacts out of git unless you intentionally want to version a tiny test fixture.
