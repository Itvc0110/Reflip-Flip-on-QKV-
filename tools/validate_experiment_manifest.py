"""Validate configs/experiment_manifest.json against the current repo."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_script(command: str) -> str | None:
    parts = shlex.split(command, posix=False)
    for part in parts:
        if part.endswith(".py"):
            return part
    return None


def check_path(root: Path, rel_path: str, problems: list[str]) -> None:
    path = root / rel_path
    if not path.exists():
        problems.append(f"Missing referenced path: {rel_path}")


def check_command_script(root: Path, command: str, problems: list[str]) -> None:
    script = command_script(command)
    if script is None:
        return
    path = root / script
    if not path.exists():
        problems.append(f"Command references missing script: {script}")


def validate_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []

    reference_workbook = manifest.get("reference_workbook")
    if isinstance(reference_workbook, str):
        check_path(root, reference_workbook, problems)
    else:
        problems.append("reference_workbook must be a string")

    tracking = manifest.get("tracking")
    if isinstance(tracking, dict):
        for key, rel_path in tracking.items():
            if isinstance(rel_path, str):
                check_path(root, rel_path, problems)
            else:
                problems.append(f"tracking.{key} must be a string")
    else:
        problems.append("tracking must be an object")

    setup = manifest.get("setup")
    if isinstance(setup, dict):
        for key in ["standalone_requirements", "full_eval_requirements"]:
            rel_path = setup.get(key)
            if isinstance(rel_path, str):
                check_path(root, rel_path, problems)
            else:
                problems.append(f"setup.{key} must be a string")
        for key in ["status_command", "standalone_preflight", "full_eval_preflight"]:
            command = setup.get(key)
            if isinstance(command, str):
                check_command_script(root, command, problems)
            else:
                problems.append(f"setup.{key} must be a string")
    else:
        problems.append("setup must be an object")

    standalone = manifest.get("standalone_qkv")
    if isinstance(standalone, dict):
        for key in ["export_command", "smoke_export_command", "run_command", "summarize_command"]:
            command = standalone.get(key)
            if isinstance(command, str):
                check_command_script(root, command, problems)
            else:
                problems.append(f"standalone_qkv.{key} must be a string")
    else:
        problems.append("standalone_qkv must be an object")

    full_model = manifest.get("full_model")
    if isinstance(full_model, dict):
        methods = full_model.get("methods")
        if isinstance(methods, list):
            for index, method in enumerate(methods):
                if not isinstance(method, dict):
                    problems.append(f"full_model.methods[{index}] must be an object")
                    continue
                script = method.get("script")
                if isinstance(script, str):
                    check_path(root, script, problems)
                else:
                    problems.append(f"full_model.methods[{index}].script must be a string")
                command = method.get("command")
                if isinstance(command, str):
                    check_command_script(root, command, problems)
                else:
                    problems.append(f"full_model.methods[{index}].command must be a string")
        else:
            problems.append("full_model.methods must be a list")

        for key in ["perplexity_compare_command", "summarize_lm_eval_command"]:
            command = full_model.get(key)
            if isinstance(command, str):
                check_command_script(root, command, problems)
            else:
                problems.append(f"full_model.{key} must be a string")
    else:
        problems.append("full_model must be an object")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment manifest references")
    parser.add_argument("--manifest", default="configs/experiment_manifest.json")
    args = parser.parse_args()

    root = Path.cwd()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 1

    manifest = load_json(manifest_path)
    problems = validate_manifest(root, manifest)

    if problems:
        print("Manifest validation failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print(f"Manifest OK: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
