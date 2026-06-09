# Smart-Flip Reference Workbook Notes

`Smart-Flip Quantization Results.xlsx` contains historical full-model results. It should be used as a comparison template, not as an input required by the standalone QKV scripts.

## Workbook Structure

Current sheets:

- `4 bit`, `3bit`: mixed method comparisons including FP/origin, RTN, RTN+Flip, AWQ, and enhanced variants.
- `RTN`, `RTN 3bit`, `RTN vs RTN+Flip 4 bit (per chan`, `RTN vs RTN+Flip 3 bit (per chan`, `tune_rtn`: RTN baselines and Flip sweeps, especially knee/flip tuning.
- `GPTQ`, `gptq_bc_4b`: GPTQ and GPTQ+Flip/+BC reference rows.
- `AdaRound`, `ada_3bit`: AdaRound and AdaRound+Flip/+BC reference rows.
- `Tune`, `tune_llama3`, `Sheet11`, `abs`, `sensitive`: tuning/ablation sheets for AWQ, EGBC/Flip, knee settings, and flip percentages.
- `Sheet13`, `Copy of RTN`, `Copy of Tune`: empty or copied/working sheets; treat as lower priority unless a row is explicitly needed.

Use this to print a compact local preview:

```bash
python tools/inspect_smart_flip_workbook.py --rows 8
```

## Metrics To Match Later

Perplexity metrics:

- `Wiki` / `wiki_ppl`: WikiText-2 perplexity. Lower is better.
- `C4`: C4 perplexity. Lower is better.
- Some sheets include a PPL average column over Wiki and C4.

lm-eval style task metrics:

- `arc-c`: ARC Challenge. Higher is better.
- `arc-e`: ARC Easy. Higher is better.
- `boolq`: BoolQ. Higher is better.
- `hellaswag`: HellaSwag. Higher is better.
- `lambada`: LAMBADA. Higher is better.
- `openbookqa`: OpenBookQA. Higher is better.
- `piqa`: PIQA. Higher is better.
- `rte`: RTE. Higher is better.
- `winogrande`: WinoGrande. Higher is better.
- `avg` / `avg5`: average across the task subset used in that sheet. Only compare averages when the task subset is identical.

## Method Labels In The Workbook

- `Origin`, `Origin (FP)`, `FP16`, `float`: unquantized baseline.
- `RTN`, `Naive RTN`: round-to-nearest baseline.
- `RTN+Flip`: round-to-nearest plus Flip correction.
- `AWQ`: activation-aware weight quantization baseline.
- `+EGBC`, `Best`, `NoKnee`, `FreeF`, `No F`, `No K`: ablation/tuned enhanced Flip-style variants.
- `GPTQ`, `GPTQ+Flip`, `AdaRound`, `AdaRound+Flip`: other quantization baselines plus Flip variants.
- `+BC`: bias/bias-correction style post-processing in the historical runs.

## Future Result Recording Policy

For new repo runs, record:

- model path and model family
- method script and method label
- bit width and group size
- calibration dataset, sample count, and sequence length
- knee tolerance and flip percent when applicable
- output model directory
- WikiText-2 and C4 perplexity
- exact lm-eval task list and task scores
- average only when all compared rows use the same tasks

Record each run in `results/experiment_ledger.jsonl`. Keep the workbook as reference rather than editing it directly.

After running `lm_eval`, use:

```bash
python tools/summarize_lm_eval.py --path ./results/lm_eval --append-ledger
```

The summarizer maps common lm-eval task ids such as `arc_challenge`, `arc_easy`, `boolq`, `piqa`, and `rte` back to workbook-style names.
