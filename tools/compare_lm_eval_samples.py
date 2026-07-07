"""Compare two lm-eval --log_samples runs and surface the questions ReFlip fixed.

Given two output directories produced by `lm_eval ... --log_samples --output_path DIR`
(a baseline, e.g. RTN, and a variant, e.g. RTN + ReFlip), this tool joins the
per-document sample logs for one multiple-choice task, reports accuracy for both
runs, and lists the documents where the baseline answered incorrectly but the
variant answered correctly (and the reverse). Optionally renders, per selected
example, a bar chart of each answer choice's log-likelihood under both models.

Only the documented lm-eval sample schema is used: `doc_id`, `doc`,
`filtered_resps` (list of [loglikelihood, is_greedy] per choice), and, when
present, the per-sample `acc` metric. The predicted choice is the argmax
log-likelihood, exactly lm-eval's own `acc` decision rule for multiple choice.

Usage:
    python tools/compare_lm_eval_samples.py \
        --baseline-dir results/lm_eval/llama3_rtn \
        --variant-dir results/lm_eval/llama3_rtn_reflip \
        --task arc_easy \
        --top-n 5 \
        --plot-dir results/lm_eval/flipped_examples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_samples_file(root: Path, task: str) -> Path:
    """Newest samples_<task>*.jsonl anywhere under root."""
    candidates = sorted(root.rglob(f"samples_{task}*.jsonl"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no samples_{task}*.jsonl under {root}")
    return candidates[-1]


def load_samples(path: Path) -> dict[int, dict]:
    samples = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            samples[int(rec["doc_id"])] = rec
    return samples


def choice_loglikelihoods(rec: dict) -> list[float]:
    """One log-likelihood per answer choice, from filtered_resps (fallback: resps)."""
    resps = rec.get("filtered_resps") or rec.get("resps")
    lls = []
    for r in resps:
        # multiple_choice entries are [ll, is_greedy] (sometimes nested one level)
        while isinstance(r, (list, tuple)) and r and isinstance(r[0], (list, tuple)):
            r = r[0]
        lls.append(float(r[0]) if isinstance(r, (list, tuple)) else float(r))
    return lls


def predicted_index(rec: dict) -> int:
    lls = choice_loglikelihoods(rec)
    return max(range(len(lls)), key=lambda i: lls[i])


def gold_index(rec: dict) -> int | None:
    """Gold choice index: from ARC-style doc (answerKey vs choices.label), else `target`."""
    doc = rec.get("doc", {})
    choices = doc.get("choices")
    if isinstance(choices, dict) and "label" in choices and "answerKey" in doc:
        try:
            return list(choices["label"]).index(doc["answerKey"])
        except ValueError:
            pass
    target = rec.get("target")
    if isinstance(target, int):
        return target
    if isinstance(target, str) and target.strip().isdigit():
        return int(target.strip())
    # last resort: per-sample acc tells us whether the prediction was the gold one
    return None


def is_correct(rec: dict) -> bool:
    gold = gold_index(rec)
    if gold is not None:
        return predicted_index(rec) == gold
    acc = rec.get("acc")
    if acc is not None:
        return bool(round(float(acc)))
    raise ValueError(f"cannot determine correctness for doc_id={rec.get('doc_id')}")


def doc_question(rec: dict) -> str:
    doc = rec.get("doc", {})
    return doc.get("question") or doc.get("query") or json.dumps(doc)[:120]


def doc_choice_texts(rec: dict) -> list[str]:
    doc = rec.get("doc", {})
    choices = doc.get("choices")
    if isinstance(choices, dict) and "text" in choices:
        return list(choices["text"])
    if isinstance(choices, list):
        return [str(c) for c in choices]
    return [f"choice {i}" for i in range(len(choice_loglikelihoods(rec)))]


def plot_example(doc_id: int, base: dict, var: dict, out_path: Path,
                 baseline_label: str, variant_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    texts = doc_choice_texts(base)
    lls_b, lls_v = choice_loglikelihoods(base), choice_loglikelihoods(var)
    gold = gold_index(base)
    n = len(texts)
    xs = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.bar(xs - width / 2, lls_b, width, color="#1f77b4", alpha=0.55, label=baseline_label)
    ax.bar(xs + width / 2, lls_v, width, color="#2ca02c", label=variant_label)
    labels = []
    for i, t in enumerate(texts):
        short = t if len(t) <= 28 else t[:25] + "..."
        labels.append(("* " if i == gold else "") + short)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("log-likelihood (higher = preferred)")
    q = doc_question(base)
    ax.set_title((q if len(q) <= 90 else q[:87] + "...") + "\n(* = correct answer)",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, help="lm-eval output dir of the baseline run")
    ap.add_argument("--variant-dir", required=True, help="lm-eval output dir of the improved run")
    ap.add_argument("--task", default="arc_easy")
    ap.add_argument("--baseline-label", default="RTN")
    ap.add_argument("--variant-label", default="RTN + ReFlip")
    ap.add_argument("--top-n", type=int, default=5, help="How many fixed examples to print")
    ap.add_argument("--plot-dir", default="", help="If set, save one bar chart per printed example")
    args = ap.parse_args()

    base_file = find_samples_file(Path(args.baseline_dir), args.task)
    var_file = find_samples_file(Path(args.variant_dir), args.task)
    print(f"baseline samples: {base_file}")
    print(f"variant  samples: {var_file}")

    base = load_samples(base_file)
    var = load_samples(var_file)
    common = sorted(set(base) & set(var))
    if not common:
        raise SystemExit("no shared doc_ids between the two runs")

    fixed, regressed, base_correct, var_correct = [], [], 0, 0
    for doc_id in common:
        b_ok, v_ok = is_correct(base[doc_id]), is_correct(var[doc_id])
        base_correct += b_ok
        var_correct += v_ok
        if not b_ok and v_ok:
            fixed.append(doc_id)
        elif b_ok and not v_ok:
            regressed.append(doc_id)

    n = len(common)
    print("\n" + "=" * 72)
    print(f"Task: {args.task}   |   shared documents: {n}")
    print(f"{args.baseline_label:<16} accuracy: {base_correct}/{n} = {base_correct / n:.4f}")
    print(f"{args.variant_label:<16} accuracy: {var_correct}/{n} = {var_correct / n:.4f}")
    print(f"fixed by {args.variant_label}: {len(fixed)}   |   regressed: {len(regressed)}")
    print("=" * 72)

    for rank, doc_id in enumerate(fixed[: args.top_n], 1):
        b, v = base[doc_id], var[doc_id]
        texts = doc_choice_texts(b)
        gold = gold_index(b)
        print(f"\n[{rank}] doc_id={doc_id}: {doc_question(b)}")
        lls_b, lls_v = choice_loglikelihoods(b), choice_loglikelihoods(v)
        pb, pv = predicted_index(b), predicted_index(v)
        for i, t in enumerate(texts):
            marks = "".join([
                " <= correct" if i == gold else "",
                f"  [{args.baseline_label} pick]" if i == pb else "",
                f"  [{args.variant_label} pick]" if i == pv else "",
            ])
            print(f"    {t[:60]:<62} ll {args.baseline_label}={lls_b[i]:8.3f}"
                  f"  {args.variant_label}={lls_v[i]:8.3f}{marks}")
        if args.plot_dir:
            out = Path(args.plot_dir) / f"fixed_{rank:02d}_doc{doc_id}.png"
            plot_example(doc_id, b, v, out, args.baseline_label, args.variant_label)
            print(f"    -> saved {out}")


if __name__ == "__main__":
    main()
