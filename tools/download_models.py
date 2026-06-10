"""Download repo models to their expected local paths.

Examples:
    python tools/download_models.py --list
    python tools/download_models.py --model llama3-8b
    python tools/download_models.py --model mistral-7b-v0.3
    python tools/download_models.py --model all
    python tools/download_models.py --model meta-llama/Meta-Llama-3-8B --local-dir ./models/Llama-3-8B

For gated models, run `huggingface-cli login` first or pass `--token <token>`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_id: str
    local_dir: str
    notes: str = ""


MODEL_SPECS = {
    "llama3-8b": ModelSpec(
        alias="llama3-8b",
        hf_id="meta-llama/Meta-Llama-3-8B",
        local_dir="./models/Llama-3-8B",
        notes="Gated model; requires Hugging Face access approval.",
    ),
    "mistral-7b-v0.3": ModelSpec(
        alias="mistral-7b-v0.3",
        hf_id="mistralai/Mistral-7B-v0.3",
        local_dir="./models/Mistral-7B-v0.3",
    ),
    "minicpm-2b": ModelSpec(
        alias="minicpm-2b",
        hf_id="openbmb/MiniCPM-2B-sft-bf16",
        local_dir="./models/MiniCPM-2B-sft-bf16",
    ),
}


DEFAULT_IGNORE_PATTERNS = [
    "original/*",
    "*.pth",
    "*.bin",
    "*.msgpack",
    "*.h5",
    "*.ot",
]


def parse_token(token: str | None, use_token: bool) -> str | bool | None:
    if token:
        return token
    if use_token:
        return True
    return None


def resolve_specs(model_arg: str, local_dir: str | None) -> list[ModelSpec]:
    if model_arg == "all":
        if local_dir:
            raise ValueError("--local-dir cannot be used with --model all")
        return list(MODEL_SPECS.values())

    if model_arg in MODEL_SPECS:
        spec = MODEL_SPECS[model_arg]
        if local_dir:
            return [ModelSpec(spec.alias, spec.hf_id, local_dir, spec.notes)]
        return [spec]

    if "/" not in model_arg:
        aliases = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model alias {model_arg!r}. Known aliases: {aliases}")
    if not local_dir:
        safe_name = model_arg.split("/")[-1]
        local_dir = f"./models/{safe_name}"
    return [ModelSpec(alias=model_arg, hf_id=model_arg, local_dir=local_dir)]


def print_model_list() -> None:
    print("Known model aliases:")
    for spec in MODEL_SPECS.values():
        print(f"  {spec.alias}")
        print(f"    hf_id:     {spec.hf_id}")
        print(f"    local_dir: {spec.local_dir}")
        if spec.notes:
            print(f"    notes:     {spec.notes}")


def download_spec(
    spec: ModelSpec,
    *,
    revision: str | None,
    token: str | bool | None,
    max_workers: int,
    all_files: bool,
    dry_run: bool,
) -> None:
    local_dir = Path(spec.local_dir)
    ignore_patterns = None if all_files else DEFAULT_IGNORE_PATTERNS

    print("=" * 80)
    print(f"Model:     {spec.alias}")
    print(f"HF id:     {spec.hf_id}")
    print(f"Local dir: {local_dir}")
    if ignore_patterns:
        print(f"Ignoring:  {', '.join(ignore_patterns)}")
    if spec.notes:
        print(f"Notes:     {spec.notes}")

    if dry_run:
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = snapshot_download(
        repo_id=spec.hf_id,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
        max_workers=max_workers,
        ignore_patterns=ignore_patterns,
    )
    print(f"Downloaded to: {downloaded_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Hugging Face models used by this repo")
    parser.add_argument(
        "--model",
        default="llama3-8b",
        help="Model alias, Hugging Face repo id, or 'all' (default: llama3-8b)",
    )
    parser.add_argument("--local-dir", help="Override local output directory for one model")
    parser.add_argument("--revision", help="Optional Hugging Face revision/commit/tag")
    parser.add_argument("--token", help="Hugging Face token string for gated/private models")
    parser.add_argument(
        "--use-token",
        action="store_true",
        help="Use cached Hugging Face token from `huggingface-cli login`",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel download workers (default: 1, safer on Windows/OneDrive)",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every file, including original checkpoints. Default downloads HF-format files only.",
    )
    parser.add_argument("--list", action="store_true", help="List known aliases and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading")
    args = parser.parse_args()

    if args.list:
        print_model_list()
        return 0

    specs = resolve_specs(args.model, args.local_dir)
    token = parse_token(args.token, args.use_token)
    for spec in specs:
        download_spec(
            spec,
            revision=args.revision,
            token=token,
            max_workers=args.max_workers,
            all_files=args.all_files,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
