"""Print a compact status snapshot for the Flip/ReFlip experiment workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> int:
    print("=" * 80)
    print(" ".join(command))
    print("=" * 80)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    print(f"\nexit code: {completed.returncode}\n")
    return int(completed.returncode)


def count_ledger_records(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Show experiment readiness status")
    parser.add_argument("--skip-preflights", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    ledger = root / "results" / "experiment_ledger.jsonl"
    workbook = root / "Smart-Flip Quantization Results.xlsx"
    manifest = root / "configs" / "experiment_manifest.json"

    print("=" * 80)
    print("Flip/ReFlip Experiment Status")
    print("=" * 80)
    print(f"Repo:     {root}")
    print(f"Workbook: {'OK' if workbook.exists() else 'MISSING'} ({workbook})")
    print(f"Manifest: {'OK' if manifest.exists() else 'MISSING'} ({manifest})")

    try:
        ledger_count = count_ledger_records(ledger)
    except Exception as exc:  # noqa: BLE001 - status command should explain parse failures.
        print(f"Ledger:   ERROR ({exc})")
        ledger_count = None
    else:
        if ledger_count is None:
            print(f"Ledger:   MISSING ({ledger})")
        else:
            print(f"Ledger:   OK ({ledger_count} records)")

    print("\nPrimary docs:")
    for rel_path in [
        "RUN_QKV_STANDALONE.md",
        "RUN_FULL_MODEL_EVAL.md",
        "EXPERIMENT_TRACKER.md",
        "MODEL_DOWNLOADS.md",
        "SMART_FLIP_REFERENCE.md",
        "configs/experiment_manifest.json",
        "tools/validate_experiment_manifest.py",
        "tools/print_experiment_commands.py",
    ]:
        path = root / rel_path
        print(f"- {rel_path}: {'OK' if path.exists() else 'MISSING'}")
    sys.stdout.flush()

    if args.skip_preflights:
        return 0

    qkv_status = run_command([sys.executable, "tools/preflight_qkv.py"])
    full_status = run_command([sys.executable, "tools/preflight_full_eval.py"])

    print("=" * 80)
    print("Status Summary")
    print("=" * 80)
    print(f"Standalone QKV preflight: {'READY' if qkv_status == 0 else 'NOT READY'}")
    print(f"Full-model/eval preflight: {'READY' if full_status == 0 else 'NOT READY'}")
    print("\nIf not ready, follow the missing-item list printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
