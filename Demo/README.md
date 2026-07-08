# Defense Demo — Flip/ReFlip on Llama-3-8B (Kaggle)

Two recorded demonstrations for the thesis defense, both runnable end-to-end on a free
Kaggle **GPU T4 x2** session:

1. **Demo 1 — standalone Q–K case study**: reproduces the thesis case study
   (Llama-3-8B, layer 8, GQA group 3): RTN → Flip → Flip + ReFlip on one GQA group,
   with the per-head error table and the Kneedle figures generated live.
2. **Demo 2 — full-model comparison**: full-precision vs `RTN` vs `RTN + ReFlip`
   (structurally RTN → Flip → ReFlip, matching the thesis full-model tables) on
   WikiText-2 perplexity and an ARC-Easy accuracy subset, plus concrete example
   questions that ReFlip fixes.

The heavy work (quantizing the 8B model) is done **once** in notebook `00` and saved
as private Kaggle Datasets; the recorded notebook `01` only loads checkpoints and runs
evaluations, so the video stays within 7–10 minutes.

> **Shortcut (recommended when defense time is limited):**
> `kaggle/02_demo_qkv_only.ipynb` is a fully **self-contained** version of Demo 1 only —
> no datasets, no notebook 00, no pre-quantized checkpoints. It downloads the model,
> then runs one repo script per cell (`xspot.py` → `fast_quantize_qkv.py` →
> summary/figures) and ends with a single cell that visualizes all results.
> Total ≈ 25 minutes on T4 x2; needs only the `HF_TOKEN` secret (or paste the token at
> the prompt in Cell 2). Present it as: "due to time constraints we demo the standalone
> QKV case study live; the full-model results are reported in the thesis."

---

## Prerequisites

- **Hugging Face token** with access to the gated `meta-llama/Meta-Llama-3-8B`
  (request access on its model page first). When a notebook reaches the model-download
  cell it **prompts you to paste the token** (input hidden). For non-interactive
  **Save & Run All** runs the prompt cannot appear — add the token as a Kaggle Secret
  named `HF_TOKEN` instead (*Add-ons → Secrets*), which is used automatically as the
  fallback.
- Kaggle account with GPU quota (~30 h/week). Budget: ~4–6 GPU-hours total for
  notebook `00` (both runs), well under one week's quota.
- This repository pushed to GitHub (public, or use a token to clone).

Repository: `https://github.com/Itvc0110/Reflip-Flip-on-QKV-.git`

---

## Step 1 — Quantize and save checkpoints (`kaggle/00_quantize_and_save.ipynb`)

Kaggle notebook output is capped at ~19.5 GB and each fp16 checkpoint is ~15 GB, so
the notebook is **parameterized and run twice**, one variant per version:

| Run | Set `VARIANT =` | Produces | Approx. time |
|-----|-----------------|----------|--------------|
| 1 | `"rtn"` | `llama3_rtn/` checkpoint (plain RTN via `rtn_xl.py`) **+ Demo-1 artifacts** (`xspot` export, `quantization_results.npz`, figures) | ~1–1.5 h |
| 2 | `"rtn_reflip"` | `llama3_rtn_reflip/` checkpoint (`rtn_gqa_xl.py --apply-gqa-reflip`: RTN base → Flip → GQA ReFlip). Runs with `--n-calib 64 --max-tokens-per-sample 256` — Kaggle's ~32 GB RAM cannot hold the full 128×512 float32 activation batches (~17 GB/batch → SIGKILL); the reduced setting is ~4.3 GB/batch. | ~1.5–2.5 h |

For each run: set `VARIANT`, **Save Version → Save & Run All (Commit)**, wait for it to
finish, then from the notebook's *Output* tab click **New Dataset** and name them
`llama3-rtn-demo` and `llama3-rtn-reflip-demo` (keep both **Private** — they contain
weights derived from gated Llama-3).

Settings that matter (already set inside the notebook):
- Accelerator: **GPU T4 x2**; Internet: **On**; Persistence: none needed.
- The base model is downloaded from HF at run time and is **not** copied into the
  output (only quantized derivatives are saved).

## Step 2 — Record the demo (`kaggle/01_demo_recording.ipynb`)

Attach as inputs: the two datasets from Step 1 + the `HF_TOKEN` secret.
Accelerator: **GPU T4 x2**, Internet On.

The notebook runs, in order (timings on T4 x2):

| Segment | What the committee sees | Time |
|---------|------------------------|------|
| 1. Setup | clone repo, attach checkpoints, paths printed | ~3 min (cut from video) |
| 2. Demo 1 | standalone pipeline output: per-head Q–K error table (RTN → Flip → Flip+ReFlip) + Kneedle figures, regenerated live from the saved npz | ~1 min |
| 3. Download | full-precision Llama-3-8B from HF (token prompt) | ~10 min (cut) |
| 4. Demo 2 — quick compare | `Demo/demo_quick_compare.py`: the SAME seeded ARC-Easy test questions scored on full precision → RTN → RTN+ReFlip (one model in memory at a time). Prints a running accuracy + s/question per model, then a **showcase**: questions RTN answers wrong and RTN+ReFlip answers right, with every choice's log-likelihood, each model's pick, and the gold answer | ~5–6 min × 3 |
| 5. Summary table | accuracy · mean s/question · generation tokens/s · peak VRAM · checkpoint size for all three models, next to the thesis Table 4.7 reference | seconds |
| 6. Wrap-up | takeaway markdown | seconds |

**Reading the time/memory columns honestly:** all three checkpoints are dense FP16
(research storage), so s/question, tokens/s, and VRAM are expected to be ~equal — the
demo's claim is "Flip/ReFlip add **zero** inference overhead while accuracy improves",
not "INT4 speedup" (a packed-INT4 build would be ~4.2 GB; kernels are out of scope).
Record segments 2→6; cut the waits. Suggested video cut (7–10 min): 0–1′ repo +
method recap, 1–4′ Demo 1, 4–8′30″ Demo 2 (showcase questions get the most airtime),
8′30″–9′30″ summary table + wrap-up.

---

## What is honest to claim in the video

- Demo 1 regenerates the thesis case-study numbers from the same pipeline
  (`xspot.py` → `fast_quantize_qkv.py`) — values match Tables 4.2/4.3.
- Demo 2 is a **fresh run**: `rtn_gqa_xl.py` is a new assembly (identity-scale
  subclass of the tested `awq_gqa_xl.py` pipeline), the ARC subset is 250 of 2,376
  questions, and lm-eval's `wikitext` word-perplexity is not the thesis's
  sliding-window protocol. Present it as "same direction as the thesis tables",
  not as a bit-exact reproduction.

## Local sanity checks (no GPU needed) — already passing

```bash
python -m py_compile rtn_gqa_xl.py tools/compare_lm_eval_samples.py
python tests/test_rtn_gqa_xl_smoke.py          # tiny GQA model end-to-end on CPU
python tests/test_compare_lm_eval_samples.py   # synthetic lm-eval fixtures
```

## Key commands (what the notebooks run under the hood)

```bash
# Demo 1 — standalone case study
python xspot.py --model-path <LLAMA3> --layer-id 8 --group-id 3 --output-dir ./xspot_layer8_group3
python fast_quantize_qkv.py --data-dir ./xspot_layer8_group3 --group-id 3 \
       --output-dir ./results/qkv_demo --critical-dim-pct 0.1
python tools/summarize_qkv_results.py --npz ./results/qkv_demo/quantization_results.npz
python tools/plot_kneedle_sensitivity.py --npz ./results/qkv_demo/quantization_results.npz --out figures/reflip_kneedle.png
python tools/plot_flip_activation_kneedle.py --npz ./results/qkv_demo/quantization_results.npz --out figures/flip_kneedle.png

# Full-model checkpoints
python rtn_xl.py     --model-path <LLAMA3> --output-dir ./quantized_models/llama3_rtn --group-size 128
python rtn_gqa_xl.py --model-path <LLAMA3> --output-dir ./quantized_models/llama3_rtn_reflip \
       --n-calib 128 --apply-gqa-reflip --gqa-critical-dim-pct 0.15 --gqa-max-flip-pct 0.05

# Demo 2 — quick 3-way compare (accuracy vs gold, per-question time, tokens/s, VRAM, disk)
python Demo/demo_quick_compare.py \
       --fp-path <LLAMA3> \
       --rtn-path ./quantized_models/llama3_rtn \
       --reflip-path ./quantized_models/llama3_rtn_reflip \
       --n-questions 100 --showcase 5 --gen-tokens 64 \
       --out-json quick_compare.json

# (optional, full-benchmark route) lm-eval per model + diff of two --log_samples runs
lm_eval --model hf --model_args pretrained=<PATH>,dtype=float16,parallelize=True \
        --tasks wikitext,arc_easy --limit 250 --batch_size 4 \
        --log_samples --output_path results/lm_eval/<variant>
python tools/compare_lm_eval_samples.py \
       --baseline-dir results/lm_eval/llama3_rtn \
       --variant-dir  results/lm_eval/llama3_rtn_reflip \
       --task arc_easy --top-n 5 --plot-dir results/lm_eval/flipped_examples
```
