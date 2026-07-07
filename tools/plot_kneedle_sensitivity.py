"""Plot the REAL ReFlip dimension-sensitivity curve and Kneedle regions.

Uses the actual standalone-run arrays in ``quantization_results.npz`` (no synthetic data):
for query head ``h`` and output dimension ``a`` the thesis dimension-sensitivity is

    S_a = |k_hat_a| * sqrt( sum_b (alpha^(h)_{a,g(b)})^2 * x_b^2 ),

where k_hat = K_quant_flip (the fixed quantized key used by ReFlip), alpha = Wq_scales_flip
expanded from 32 groups to 4096 input columns (128 columns per group), and x = X (the
James-Stein representative activation). The sorted, unit-normalized curve is partitioned by the
Kneedle knee into a protected extreme region [1..rho], a moderate region of H_R = floor(0.1*d_h)
dimensions immediately after the knee, and a low-priority tail.

Output: Report_Thesis/Figure/reflip_sensitivity_kneedle.png
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

PROTECTED = "#f4cccc"   # light red
MODERATE = "#ffe599"    # gold
LOW = "#cfe2f3"         # light blue
CURVE = "#1f3b57"       # dark navy
KNEE = "#cc0000"        # red


def dimension_sensitivity(npz, head: int) -> np.ndarray:
    x = npz["X"].astype(np.float64)                       # (d_in,)
    khat = npz["K_quant_flip"].astype(np.float64)         # (d_h,)  fixed quantized key
    scales = npz["Wq_scales_flip"][head].astype(np.float64)  # (d_h, n_groups)
    d_h, n_groups = scales.shape
    d_in = x.shape[0]
    alpha = np.repeat(scales, d_in // n_groups, axis=1)[:, :d_in]  # (d_h, d_in)
    return np.abs(khat) * np.sqrt((alpha**2 * x[None, :] ** 2).sum(axis=1))  # (d_h,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="quantization_results.npz")
    ap.add_argument("--head", type=int, default=1)
    ap.add_argument("--moderate-pct", type=float, default=0.1)
    ap.add_argument("--out", default="Report_Thesis/Figure/reflip_sensitivity_kneedle.png")
    args = ap.parse_args()

    npz = np.load(args.npz, allow_pickle=True)
    s = dimension_sensitivity(npz, args.head)
    d_h = s.shape[0]
    s_sorted = np.sort(s)[::-1]                           # decreasing
    rho = int(find_knee_point(s_sorted.copy()))           # knee index (0-based)
    h_r = max(int(args.moderate_pct * d_h), 1)            # H_R = floor(0.1 d_h)

    u = np.arange(d_h) / (d_h - 1)                        # normalized rank in [0,1]
    y = (s_sorted - s_sorted.min()) / (s_sorted.max() - s_sorted.min())  # normalized [0,1]

    u_knee = u[rho]
    u_mod_end = u[min(rho + h_r, d_h - 1)]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axvspan(0, u_knee, color=PROTECTED, label=f"protected extreme  (dims 1-{rho+1})")
    ax.axvspan(u_knee, u_mod_end, color=MODERATE,
               label=f"moderate candidates  ($H_R={h_r}$ dims)")
    ax.axvspan(u_mod_end, 1.0, color=LOW, label="low-priority tail")
    ax.plot(u, y, color=CURVE, lw=2.4, zorder=3, label=r"sorted sensitivity $y_t$")
    ax.scatter([u_knee], [y[rho]], color=KNEE, s=70, zorder=5)
    ax.annotate("detected knee\n(included in protected region)",
                xy=(u_knee, y[rho]), xytext=(u_knee + 0.16, y[rho] + 0.22),
                color=KNEE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=KNEE, lw=1.4))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"normalized query-dimension rank $u_t$")
    ax.set_ylabel(r"normalized dimension sensitivity $y_t$")
    ax.set_title("ReFlip dimension-sensitivity $\\mathcal{S}_a$ and Kneedle regions\n"
                 "(Llama-3-8B, layer 8, GQA group 3, query head %d; real run data)" % args.head)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"knee rho={rho} (u={u_knee:.3f}); moderate dims={h_r}; saved {out}")


if __name__ == "__main__":
    main()
