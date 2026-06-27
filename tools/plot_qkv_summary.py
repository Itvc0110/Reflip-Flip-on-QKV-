"""Generate report plots from a standalone scalar Q--K surrogate summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_NAMES = (
    "score_comparison_by_head.png",
    "error_waterfall_by_head.png",
    "method_summary_errors.png",
    "flip_cost_vs_error_reduction.png",
    "manual_attention_error_by_head.png",
)

METHOD_COLORS = {
    "Original": "#4c4c4c",
    "Nearest": "#1f77b4",
    "Flip": "#ff7f0e",
    "ReFlip": "#2ca02c",
}


def _load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("heads"):
        raise ValueError(f"No head results found in {summary_path}")
    return summary


def report_method_label(method: str) -> str:
    return {
        "Original": "Full precision",
        "Nearest": "RTN",
        "Flip": "Flip",
        "ReFlip": "Flip + ReFlip",
    }[method]


def report_title_prefix(summary: dict) -> str:
    xspot = summary.get("xspot", {})
    layer = xspot.get("layer_id", "?")
    group = xspot.get("group_id", "?")
    return f"Standalone Q--K Surrogate: Layer {layer}, GQA Group {group}"


def _head_labels(heads: Iterable[dict]) -> list[str]:
    return [f"Head {head['head']}" for head in heads]


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_comparison(summary: dict, output_dir: Path) -> Path:
    heads = summary["heads"]
    labels = _head_labels(heads)
    x = np.arange(len(heads))
    width = 0.19

    series = [
        ("Original", [head["original_score"] for head in heads]),
        ("Nearest", [head["nearest_score"] for head in heads]),
        ("Flip", [head["flip_score"] for head in heads]),
        ("ReFlip", [head["reflip_score"] for head in heads]),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    offsets = (-1.5, -0.5, 0.5, 1.5)
    for offset, (name, values) in zip(offsets, series):
        ax.bar(
            x + offset * width,
            values,
            width,
            label=report_method_label(name),
            color=METHOD_COLORS[name],
            alpha=0.88,
        )

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title(f"{report_title_prefix(summary)}\nFull-Precision and Quantized Scores")
    ax.set_ylabel("Scalar Q--K surrogate score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="best")
    return _save(fig, output_dir / "score_comparison_by_head.png")


def plot_error_waterfall(summary: dict, output_dir: Path) -> Path:
    heads = summary["heads"]
    n_heads = len(heads)
    n_cols = 2 if n_heads > 1 else 1
    n_rows = int(np.ceil(n_heads / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"{report_title_prefix(summary)}\nError Reduction Waterfall by Head", y=1.02)

    for ax, head in zip(axes.ravel(), heads):
        nearest = abs(head["nearest_abs_error"])
        flip = abs(head["flip_abs_error"])
        reflip = abs(head["reflip_abs_error"])
        flip_delta = flip - nearest
        reflip_delta = reflip - flip

        labels = ["RTN", "Flip gain", "ReFlip-stage gain", "Flip + ReFlip"]
        colors = [
            METHOD_COLORS["Nearest"],
            "#2ca02c" if flip_delta <= 0 else "#d62728",
            "#2ca02c" if reflip_delta <= 0 else "#d62728",
            METHOD_COLORS["ReFlip"],
        ]

        ax.bar(0, nearest, color=colors[0], alpha=0.88)
        ax.bar(1, flip_delta, bottom=nearest, color=colors[1], alpha=0.88)
        ax.bar(2, reflip_delta, bottom=flip, color=colors[2], alpha=0.88)
        ax.bar(3, reflip, color=colors[3], alpha=0.88)

        ax.plot([0, 1], [nearest, nearest], color="#777777", linewidth=0.8, alpha=0.7)
        ax.plot([1, 2], [flip, flip], color="#777777", linewidth=0.8, alpha=0.7)
        ax.plot([2, 3], [reflip, reflip], color="#777777", linewidth=0.8, alpha=0.7)

        ax.set_title(f"Head {head['head']}")
        ax.set_ylabel("Absolute scalar Q--K surrogate error")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)

        for xpos, value in enumerate((nearest, flip_delta, reflip_delta, reflip)):
            if xpos in (1, 2):
                label = f"{value:+.4f}"
                ypos = nearest + value / 2 if xpos == 1 else flip + value / 2
            else:
                label = f"{value:.4f}"
                ypos = value
            ax.annotate(
                label,
                xy=(xpos, ypos),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for ax in axes.ravel()[len(heads) :]:
        ax.axis("off")

    return _save(fig, output_dir / "error_waterfall_by_head.png")


def plot_error_by_head(summary: dict, output_dir: Path) -> Path:
    heads = summary["heads"]
    labels = _head_labels(heads)
    x = np.arange(len(heads))
    width = 0.24
    series = [
        ("Nearest", [abs(head["nearest_abs_error"]) for head in heads]),
        ("Flip", [abs(head["flip_abs_error"]) for head in heads]),
        ("ReFlip", [abs(head["reflip_abs_error"]) for head in heads]),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for offset, (method, values) in zip((-1, 0, 1), series):
        ax.bar(
            x + offset * width,
            values,
            width,
            label=report_method_label(method),
            color=METHOD_COLORS[method],
            alpha=0.88,
        )

    ax.set_title(f"{report_title_prefix(summary)}\nAbsolute Error by Query Head")
    ax.set_ylabel("Absolute scalar Q--K surrogate error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    return _save(fig, output_dir / "manual_attention_error_by_head.png")


def plot_method_summary_errors(summary: dict, output_dir: Path) -> Path:
    metrics = summary["metrics"]
    methods = ["Nearest", "Flip", "ReFlip"]
    colors = [METHOD_COLORS[method] for method in methods]

    panels = [
        (
            "Mean absolute error",
            [
                metrics["nearest_mean_abs_attention_error"],
                metrics["flip_mean_abs_attention_error"],
                metrics["reflip_mean_abs_attention_error"],
            ],
            "Scalar Q--K surrogate error",
        ),
        (
            "Mean relative error",
            [
                metrics["nearest_mean_abs_relative_error_pct"],
                metrics["flip_mean_abs_relative_error_pct"],
                metrics["reflip_mean_abs_relative_error_pct"],
            ],
            "Relative error (%)",
        ),
        (
            "Max absolute error",
            [
                metrics["nearest_max_abs_attention_error"],
                metrics["flip_max_abs_attention_error"],
                metrics["reflip_max_abs_attention_error"],
            ],
            "Scalar Q--K surrogate error",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(f"{report_title_prefix(summary)}\nMethod-Level Error Summary", y=1.03)
    for ax, (title, values, ylabel) in zip(axes, panels):
        ax.bar([report_method_label(method) for method in methods], values, color=colors, alpha=0.88)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            ax.annotate(
                f"{value:.4f}",
                xy=(idx, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    return _save(fig, output_dir / "method_summary_errors.png")


def plot_flip_cost_vs_error_reduction(summary: dict, output_dir: Path) -> Path:
    metrics = summary["metrics"]
    flip_stats = summary.get("flip_stats", {})
    reflip_stats = summary.get("reflip_stats", {})

    methods = ["Nearest", "Flip", "ReFlip"]
    mean_errors = [
        metrics["nearest_mean_abs_attention_error"],
        metrics["flip_mean_abs_attention_error"],
        metrics["reflip_mean_abs_attention_error"],
    ]
    cumulative_operations = [
        0,
        flip_stats.get("total_flip_weights", 0),
        flip_stats.get("total_flip_weights", 0) + reflip_stats.get("total_integer_flips", 0),
    ]
    reductions = [
        0.0,
        metrics["nearest_to_flip_mean_improvement_pct"],
        metrics["nearest_to_reflip_mean_improvement_pct"],
    ]

    x = np.arange(len(methods))
    fig, ax_error = plt.subplots(figsize=(10, 5.8))
    bars = ax_error.bar(
        x,
        mean_errors,
        width=0.52,
        color=[METHOD_COLORS[method] for method in methods],
        alpha=0.86,
        label="Mean abs error",
    )
    ax_error.set_title(f"{report_title_prefix(summary)}\nCorrection Operations vs Error Reduction")
    ax_error.set_ylabel("Mean absolute scalar Q--K surrogate error")
    ax_error.set_xticks(x)
    ax_error.set_xticklabels([report_method_label(method) for method in methods])
    ax_error.grid(axis="y", alpha=0.25)

    ax_flips = ax_error.twinx()
    ax_flips.scatter(
        x,
        cumulative_operations,
        color="#111111",
        s=90,
        marker="D",
        label="Cumulative one-level operations",
        zorder=5,
    )
    ax_flips.set_ylabel("Cumulative one-level operations")
    max_operations = max(cumulative_operations) if cumulative_operations else 0
    if max_operations > 0:
        ax_flips.set_ylim(bottom=-max_operations * 0.08, top=max_operations * 1.28)

    for idx, (bar, error, flips, reduction) in enumerate(
        zip(bars, mean_errors, cumulative_operations, reductions)
    ):
        ax_error.annotate(
            f"{error:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, error),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        flip_label_offset = 12 if flips == 0 else -28
        flip_label_va = "bottom" if flips == 0 else "top"
        ax_flips.annotate(
            f"{flips:,} operations\n{reduction:.1f}% reduction",
            xy=(idx, flips),
            xytext=(0, flip_label_offset),
            textcoords="offset points",
            ha="center",
            va=flip_label_va,
            fontsize=8,
            color="#111111",
        )

    handles_a, labels_a = ax_error.get_legend_handles_labels()
    handles_b, labels_b = ax_flips.get_legend_handles_labels()
    ax_error.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    return _save(fig, output_dir / "flip_cost_vs_error_reduction.png")


def plot_summary(summary_path: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    summary_path = Path(summary_path)
    output_dir = Path(output_dir) if output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(summary_path)

    return [
        plot_score_comparison(summary, output_dir),
        plot_error_waterfall(summary, output_dir),
        plot_method_summary_errors(summary, output_dir),
        plot_flip_cost_vs_error_reduction(summary, output_dir),
        plot_error_by_head(summary, output_dir),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report plots from a standalone scalar Q--K surrogate summary."
    )
    parser.add_argument(
        "--summary",
        default="results/qkv_llama3_layer8_group3_full/manual_summary.json",
        help="Path to manual_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs. Defaults to the summary file directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written = plot_summary(args.summary, args.output_dir)
    print("Saved plots:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
