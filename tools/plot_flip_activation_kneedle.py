"""Plot the REAL Flip activation-magnitude curve and its Kneedle outlier mask.

Uses the actual standalone-run array in ``quantization_results.npz`` (no synthetic
data): x = X is the James-Stein representative activation (d_in = 4096 channels).
Flip's eligibility mask replicates ``compute_dynamic_outlier_threshold`` in
``utils_qkv.py``: sort |x_b| in DESCENDING order, run Kneedle on the first half,
take the value at the detected knee as the threshold, and freeze every input
column with |x_b| above it. Frozen columns are never flipped because a single
move there contributes x_b * delta large enough to dominate the row residual.

Output: figures/flip_activation_kneedle.png (see --out)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_qkv import find_knee_point

FROZEN = "#f4cccc"      # light red  (activation outliers, columns frozen)
ELIGIBLE = "#d9ead3"    # light green (columns Flip may act on)
CURVE = "#1f3b57"       # dark navy
KNEE = "#cc0000"        # red


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="quantization_results.npz")
    ap.add_argument("--out", default="figures/flip_activation_kneedle.png")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    x = np.abs(npz["X"].astype(np.float64))               # (d_in,)
    d_in = x.shape[0]

    # exact replication of compute_dynamic_outlier_threshold (utils_qkv.py)
    x_sorted = np.sort(x)[::-1]                           # descending
    first_half = x_sorted[: d_in // 2]
    knee_idx = int(find_knee_point(first_half))
    threshold = x_sorted[knee_idx]
    n_frozen = int((x > threshold).sum())
    pct_frozen = 100.0 * n_frozen / d_in

    u = np.arange(d_in) / (d_in - 1)                      # normalized rank
    y = (x_sorted - x_sorted.min()) / (x_sorted.max() - x_sorted.min())
    u_knee = u[knee_idx]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axvspan(0, u_knee, color=FROZEN,
               label=f"activation outliers — frozen ({n_frozen} of {d_in} "
                     f"columns, {pct_frozen:.1f}%)")
    ax.axvspan(u_knee, 1.0, color=ELIGIBLE,
               label="eligible for Flip (one-level moves allowed)")
    ax.plot(u, y, color=CURVE, lw=2.4, zorder=3,
            label=r"sorted activation magnitude $|x_b|$ (normalized)")
    ax.scatter([u_knee], [y[knee_idx]], color=KNEE, s=70, zorder=5)

    # zoom inset: the knee is a thin sliver of the 4096 channels
    axins = ax.inset_axes([0.45, 0.42, 0.5, 0.46])
    zoom_n = max(4 * knee_idx, 32)
    axins.axvspan(0, knee_idx, color=FROZEN)
    axins.axvspan(knee_idx, zoom_n, color=ELIGIBLE)
    axins.plot(np.arange(zoom_n), y[:zoom_n], color=CURVE, lw=2.0)
    axins.scatter([knee_idx], [y[knee_idx]], color=KNEE, s=50, zorder=5)
    axins.annotate(f"detected knee → threshold (ch. {knee_idx})",
                   xy=(knee_idx, y[knee_idx]),
                   xytext=(knee_idx + 0.28 * zoom_n, y[knee_idx] + 0.35),
                   color=KNEE, fontsize=9,
                   arrowprops=dict(arrowstyle="->", color=KNEE, lw=1.3))
    axins.set_title(f"zoom: first {zoom_n} of {d_in} channels", fontsize=9)
    axins.set_xlim(0, zoom_n)
    axins.tick_params(labelsize=8)
    axins.grid(True, alpha=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("normalized input-channel rank")
    ax.set_ylabel(r"normalized $|x_b|$")
    ax.set_title("Flip activation-outlier mask: Kneedle on sorted $|x_b|$\n"
                 "(James–Stein representative, Llama-3-8B layer 8; "
                 "real run data)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", framealpha=0.95, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"knee idx={knee_idx} (u={u_knee:.4f}); threshold={threshold:.4f}; "
          f"frozen={n_frozen}/{d_in} ({pct_frozen:.2f}%); saved {out}")


if __name__ == "__main__":
    main()
