"""Print runnable commands from configs/experiment_manifest.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def print_block(title: str, commands: list[str]) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
    for command in commands:
        print(command)
    print()


def standalone_commands(manifest: dict[str, Any]) -> list[str]:
    setup = manifest["setup"]
    standalone = manifest["standalone_qkv"]
    return [
        setup["status_command"],
        setup["standalone_preflight"],
        standalone["smoke_export_command"],
        standalone["export_command"],
        standalone["run_command"],
        standalone["summarize_command"],
    ]


def full_model_commands(manifest: dict[str, Any]) -> list[str]:
    setup = manifest["setup"]
    full_model = manifest["full_model"]
    commands = [
        "python -m pip install -r requirements-full-eval.txt",
        setup["full_eval_preflight"],
    ]
    commands.extend(method["command"] for method in full_model["methods"])
    commands.append(full_model["perplexity_compare_command"])
    commands.append(full_model["summarize_lm_eval_command"])
    return commands


def download_commands(manifest: dict[str, Any]) -> list[str]:
    models = manifest["models"]
    commands = ["python -m pip install huggingface_hub", "huggingface-cli login"]
    for model_info in models.values():
        command = model_info.get("download_example") or model_info.get("local_download_example")
        if command:
            commands.append(command)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Print commands from experiment manifest")
    parser.add_argument("--manifest", default="configs/experiment_manifest.json")
    parser.add_argument(
        "--section",
        choices=["all", "download", "standalone", "full-model"],
        default="all",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    manifest = load_manifest(manifest_path)

    if args.section in ["all", "download"]:
        print_block("Download / Setup Commands", download_commands(manifest))
    if args.section in ["all", "standalone"]:
        print_block("Standalone QKV Commands", standalone_commands(manifest))
    if args.section in ["all", "full-model"]:
        print_block("Full-Model / Evaluation Commands", full_model_commands(manifest))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
