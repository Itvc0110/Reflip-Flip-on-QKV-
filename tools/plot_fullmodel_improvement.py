"""Full-model five-task average: each 4-bit baseline vs its +ReFlip variant.

The values are the five-task averages avg=(arc-c+arc-e+boolq+piqa+rte)/5 reported in the thesis
Tables 4.4-4.7 (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B, Llama-3-8B), which trace to
``Smart-Flip Quantization Results.xlsx`` (sheet ``4 bit``). They are reproduced here verbatim so the
figure is guaranteed identical to the tables; no value is invented. The figure shows that every
``+ ReFlip`` variant raises the five-task average above its named RTN/GPTQ/AWQ baseline.

Output: Report_Thesis/Figure/fullmodel_reflip_improvement.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# model -> (full-precision avg, {family: (baseline_avg, reflip_avg)})  -- from Tables 4.4-4.7
DATA = {
    "Mistral-7B": (0.7268, {"RTN": (0.7104, 0.7141), "GPTQ": (0.7136, 0.7156), "AWQ": (0.7144, 0.7200)}),
    "Llama-3.1-8B": (0.7352, {"RTN": (0.7028, 0.7089), "GPTQ": (0.7096, 0.7126), "AWQ": (0.7088, 0.7166)}),
    "Qwen2.5-7B": (0.7458, {"RTN": (0.7254, 0.7279), "GPTQ": (0.7178, 0.7236), "AWQ": (0.7364, 0.7426)}),
    "Llama-3-8B": (0.7220, {"RTN": (0.6924, 0.7033), "GPTQ": (0.7090, 0.7110), "AWQ": (0.7008, 0.7094)}),
}
FAMILY_COLOR = {"RTN": "#1f77b4", "GPTQ": "#ff7f0e", "AWQ": "#2ca02c"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Report_Thesis/Figure/fullmodel_reflip_improvement.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 4.9))
    families = ["RTN", "GPTQ", "AWQ"]
    width = 0.38
    for ax, (model, (fp, fam)) in zip(axes.ravel(), DATA.items()):
        xs = np.arange(len(families))
        base = [fam[f][0] for f in families]
        refl = [fam[f][1] for f in families]
        for i, f in enumerate(families):
            ax.bar(xs[i] - width / 2, base[i], width, color=FAMILY_COLOR[f], alpha=0.45,
                   label="baseline" if i == 0 else None)
            ax.bar(xs[i] + width / 2, refl[i], width, color=FAMILY_COLOR[f],
                   label="+ ReFlip" if i == 0 else None)
            ax.annotate(f"+{(refl[i]-base[i])*100:.2f}", xy=(xs[i] + width / 2, refl[i]),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8,
                        color="#222")
        ax.axhline(fp, ls="--", lw=1.2, color="#888")
        ax.text(len(families) - 0.5, fp, "  full precision", va="center", ha="left",
                fontsize=8, color="#666")
        lo = min(base) - 0.012
        ax.set_ylim(lo, max(fp, max(refl)) + 0.006)
        ax.set_xticks(xs)
        ax.set_xticklabels(families)
        ax.set_title(model, fontsize=11)
        ax.set_ylabel("five-task average")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
