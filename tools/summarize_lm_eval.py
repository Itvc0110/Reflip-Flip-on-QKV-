"""Summarize lm-eval JSON outputs and optionally append to the experiment ledger.

This parser is intentionally tolerant of common lm-eval output layouts. It
expects a JSON file containing a top-level `results` object where each task maps
to metric values such as `acc,none`, `acc_norm,none`, or `exact_match,none`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PREFERRED_METRICS = [
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "perplexity,none",
    "word_perplexity,none",
    "byte_perplexity,none",
]

WORKBOOK_TASK_ALIASES = {
    "arc_challenge": "arc-c",
    "arc_easy": "arc-e",
    "boolq": "boolq",
    "hellaswag": "hellaswag",
    "lambada_openai": "lambada",
    "lambada": "lambada",
    "openbookqa": "openbookqa",
    "piqa": "piqa",
    "rte": "rte",
    "winogrande": "winogrande",
}


def find_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def choose_metric(task_result: dict[str, Any]) -> tuple[str, float] | None:
    for key in PREFERRED_METRICS:
        value = task_result.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    for key, value in task_result.items():
        if key.endswith("_stderr") or key.endswith(",stderr"):
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def summarize_file(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, dict):
        raise KeyError(f"No top-level 'results' object found in {path}")

    tasks: dict[str, Any] = {}
    workbook_scores: dict[str, float] = {}

    for task_name, task_result in sorted(results.items()):
        if not isinstance(task_result, dict):
            continue
        chosen = choose_metric(task_result)
        if chosen is None:
            continue
        metric_name, score = chosen
        tasks[task_name] = {
            "metric": metric_name,
            "score": score,
        }
        workbook_name = WORKBOOK_TASK_ALIASES.get(task_name)
        if workbook_name:
            workbook_scores[workbook_name] = score

    average = None
    if workbook_scores:
        average = sum(workbook_scores.values()) / len(workbook_scores)

    return {
        "file": str(path),
        "tasks": tasks,
        "workbook_scores": workbook_scores,
        "workbook_score_avg": average,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 80)
    print("lm-eval Result Summary")
    print("=" * 80)
    print(f"File: {summary['file']}")

    tasks = summary["tasks"]
    if not tasks:
        print("No numeric task metrics found.")
        return

    print("\nTasks:")
    for task_name, info in tasks.items():
        print(f"  {task_name:<20} {info['metric']:<20} {info['score']:.6f}")

    workbook_scores = summary["workbook_scores"]
    if workbook_scores:
        print("\nWorkbook-style scores:")
        for task_name, score in sorted(workbook_scores.items()):
            print(f"  {task_name:<12} {score:.6f}")
        print(f"  {'avg':<12} {summary['workbook_score_avg']:.6f}")


def append_ledger(summaries: list[dict[str, Any]], ledger_path: Path, label: str | None) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        row = {
            "status": "completed",
            "script": "tools/summarize_lm_eval.py",
            "phase": "lm_eval_summary",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            **summary,
        }
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize lm-eval JSON result files")
    parser.add_argument(
        "--path",
        default="./results/lm_eval",
        help="lm-eval JSON file or directory containing JSON outputs",
    )
    parser.add_argument("--label", default=None, help="Optional run label for ledger rows")
    parser.add_argument("--append-ledger", action="store_true")
    parser.add_argument("--ledger", default="results/experiment_ledger.jsonl")
    args = parser.parse_args()

    json_files = find_json_files(Path(args.path))
    if not json_files:
        print(f"No lm-eval JSON files found at: {args.path}")
        print("Run lm_eval first, then re-run this summarizer.")
        return 1

    summaries = []
    for path in json_files:
        try:
            summary = summarize_file(path)
        except Exception as exc:  # noqa: BLE001 - report and continue across files.
            print(f"Skipping {path}: {exc}")
            continue
        summaries.append(summary)
        print_summary(summary)
        print()

    if not summaries:
        print("No valid lm-eval result files could be summarized.")
        return 1

    if args.append_ledger:
        append_ledger(summaries, Path(args.ledger), args.label)
        print(f"Appended {len(summaries)} summary row(s) to: {args.ledger}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
