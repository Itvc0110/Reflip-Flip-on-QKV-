"""Quick 3-way demo: full precision vs RTN vs RTN+ReFlip on the same ARC-Easy questions.

For each model (loaded sequentially, freed in between) this script:
  1. scores a seeded pool of ARC-Easy TEST questions by log-likelihood over the
     answer choices (the same decision rule lm-eval uses for the benchmark),
     timing every question (CUDA-synchronized wall clock);
  2. measures short-generation speed (tokens/s) on one fixed prompt.

It then prints, for the committee:
  - a SHOWCASE table: the questions answered wrongly by RTN but correctly by
    RTN+ReFlip (and by full precision) — each with all choices, every model's
    pick, and the gold answer;
  - a SUMMARY table: accuracy on the pool, mean/median per-question inference
    time, and generation tokens/s per model.

Note on timing: all three checkpoints are dense FP16 research checkpoints
(dequantized storage), so per-question times are expected to be ~equal — the
honest reading is "Flip/ReFlip add ZERO inference overhead", not "INT4 speedup"
(the thesis makes no runtime claim; packed-INT4 kernels are out of scope).

Usage:
    python Demo/demo_quick_compare.py \
        --fp-path /kaggle/tmp/models/Llama-3-8B \
        --rtn-path /kaggle/input/llama3-rtn-demo/llama3_rtn \
        --reflip-path /kaggle/input/llama3-rtn-reflip-demo \
        --n-questions 100 --showcase 5 --out-json /kaggle/working/quick_compare.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_questions(n_questions: int, seed: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_questions, len(ds))))
    questions = []
    for rec in ds:
        labels = list(rec["choices"]["label"])
        if rec["answerKey"] not in labels:
            continue
        questions.append({
            "id": rec["id"],
            "question": rec["question"],
            "choices": list(rec["choices"]["text"]),
            "labels": labels,
            "gold": labels.index(rec["answerKey"]),
        })
    return questions


@torch.no_grad()
def choice_loglikelihoods(model, tokenizer, question: str, choices: list[str], device):
    """Sum of token log-probs of each answer continuation, batched over choices."""
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    seqs, cont_lens = [], []
    for choice in choices:
        cont_ids = tokenizer(" " + choice, return_tensors="pt",
                             add_special_tokens=False).input_ids[0]
        seqs.append(torch.cat([prompt_ids, cont_ids]))
        cont_lens.append(len(cont_ids))
    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = s
        mask[i, :len(s)] = 1
    logits = model(input_ids=batch.to(device), attention_mask=mask.to(device),
                   use_cache=False).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)
    out = []
    for i, s in enumerate(seqs):
        n, c = len(s), cont_lens[i]
        # log P(token_t | tokens_<t) for the continuation positions
        tgt = batch[i, n - c:n].to(device)
        lp = logprobs[i, n - c - 1:n - 1, :].gather(-1, tgt.unsqueeze(-1)).sum()
        out.append(lp.item())
    return out


@torch.no_grad()
def generation_speed(model, tokenizer, device, n_tokens: int):
    prompt = "The theory of relativity states that"
    ids = tokenizer(prompt, return_tensors="pt").to(device)
    model.generate(**ids, max_new_tokens=4, do_sample=False)  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=n_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    produced = out.shape[1] - ids["input_ids"].shape[1]
    return produced / dt


def checkpoint_disk_gb(path: str) -> float:
    import glob
    import os
    files = glob.glob(os.path.join(path, "*.safetensors")) or \
        glob.glob(os.path.join(path, "*.bin"))
    return sum(os.path.getsize(f) for f in files) / 1e9


def evaluate_model(label: str, path: str, questions, gen_tokens: int):
    print(f"\n{'=' * 72}\nLoading {label}: {path}\n{'=' * 72}")
    for d in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(d)
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    records, times = [], []
    for i, q in enumerate(questions):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lls = choice_loglikelihoods(model, tokenizer, q["question"], q["choices"], device)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pred = max(range(len(lls)), key=lambda j: lls[j])
        records.append({"pred": pred, "lls": lls, "time_s": dt})
        times.append(dt)
        if (i + 1) % 20 == 0:
            acc = sum(r["pred"] == qq["gold"] for r, qq in zip(records, questions)) / (i + 1)
            print(f"  {i + 1}/{len(questions)} questions | running acc {acc:.3f} "
                  f"| {statistics.mean(times):.3f}s/question")

    tps = generation_speed(model, tokenizer, device, gen_tokens)
    vram_gb = sum(torch.cuda.max_memory_allocated(d)
                  for d in range(torch.cuda.device_count())) / 1e9
    disk_gb = checkpoint_disk_gb(path)
    acc = sum(r["pred"] == q["gold"] for r, q in zip(records, questions)) / len(questions)
    print(f"  -> {label}: accuracy {acc:.4f} | mean {statistics.mean(times):.3f}s/question "
          f"| generation {tps:.1f} tokens/s | peak VRAM {vram_gb:.1f} GB "
          f"| checkpoint {disk_gb:.1f} GB")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"accuracy": acc, "mean_time_s": statistics.mean(times),
            "median_time_s": statistics.median(times), "gen_tokens_per_s": tps,
            "peak_vram_gb": vram_gb, "checkpoint_disk_gb": disk_gb,
            "records": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-path", required=True)
    ap.add_argument("--rtn-path", required=True)
    ap.add_argument("--reflip-path", required=True)
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--showcase", type=int, default=5)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="quick_compare.json")
    args = ap.parse_args()

    questions = load_questions(args.n_questions, args.seed)
    print(f"Loaded {len(questions)} ARC-Easy test questions (seed {args.seed})")

    variants = [("Full precision", args.fp_path),
                ("RTN", args.rtn_path),
                ("RTN + ReFlip", args.reflip_path)]
    results = {label: evaluate_model(label, path, questions, args.gen_tokens)
               for label, path in variants}

    # ---- showcase: RTN wrong, ReFlip right (prefer FP also right) ----
    fixed = []
    for i, q in enumerate(questions):
        rtn_ok = results["RTN"]["records"][i]["pred"] == q["gold"]
        ref_ok = results["RTN + ReFlip"]["records"][i]["pred"] == q["gold"]
        fp_ok = results["Full precision"]["records"][i]["pred"] == q["gold"]
        if not rtn_ok and ref_ok:
            fixed.append((not fp_ok, i))  # FP-correct examples first
    fixed.sort()
    showcase = [i for _, i in fixed[: args.showcase]]

    print(f"\n{'=' * 72}\nSHOWCASE — questions RTN gets WRONG and RTN+ReFlip gets RIGHT "
          f"({len(fixed)} found, showing {len(showcase)})\n{'=' * 72}")
    for rank, i in enumerate(showcase, 1):
        q = questions[i]
        print(f"\n[{rank}] {q['question']}")
        for j, text in enumerate(q["choices"]):
            marks = []
            if j == q["gold"]:
                marks.append("GOLD")
            for label in results:
                if results[label]["records"][i]["pred"] == j:
                    marks.append(label)
            lls = " ".join(f"{label.split()[0]}={results[label]['records'][i]['lls'][j]:7.2f}"
                           for label in results)
            print(f"    {q['labels'][j]}. {text[:58]:<60} ll: {lls}"
                  f"   {'<-- ' + ', '.join(marks) if marks else ''}")

    print(f"\n{'=' * 72}\nSUMMARY ({len(questions)} ARC-Easy test questions)\n{'=' * 72}")
    print(f"{'Model':<18}{'Accuracy':>10}{'Mean s/q':>10}{'Gen tok/s':>11}"
          f"{'Peak VRAM GB':>14}{'Disk GB':>9}")
    for label in results:
        r = results[label]
        print(f"{label:<18}{r['accuracy']:>10.4f}{r['mean_time_s']:>10.3f}"
              f"{r['gen_tokens_per_s']:>11.1f}{r['peak_vram_gb']:>14.1f}"
              f"{r['checkpoint_disk_gb']:>9.1f}")
    print("\nNote: all three checkpoints are stored as dense FP16 (research format), so")
    print("time and memory are expected to be ~EQUAL across the three models. The honest")
    print("takeaways: (1) Flip/ReFlip add ZERO inference overhead vs RTN; (2) accuracy")
    print("differs while cost does not. A deployable INT4-packed build of the quantized")
    print("models would need ~4.2 GB disk / VRAM for weights (4.16 effective bits per")
    print("weight incl. group metadata) - packed kernels are out of the thesis scope.")

    payload = {"questions": [{k: q[k] for k in ("id", "question", "choices", "labels", "gold")}
                             for q in questions],
               "results": results, "showcase_indices": showcase,
               "n_fixed_by_reflip": len(fixed)}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
