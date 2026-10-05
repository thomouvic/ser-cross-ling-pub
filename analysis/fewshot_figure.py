"""Clean paper figure for cross-lingual K-shot transfer.

Single panel: mean accuracy across 12 source-target pairs vs K, with shaded
+/-1 std band, one curve per model. Replaces the busier per-pair facet plot.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_COLORS = {
    "hubert":      "#1f4e79",  # navy
    "wavlm":       "#2e7d6b",  # teal
    "emotion2vec": "#c64241",  # coral
}
MODEL_LABEL = {
    "hubert":      "HuBERT",
    "wavlm":       "WavLM",
    "emotion2vec": "emotion2vec_base",
}


def main():
    # Speaker-disjoint few-shot is the primary protocol (shots drawn from a
    # held-out pool of target speakers, disjoint from the test speakers).
    disjoint_csv = RESULTS_DIR / "fewshot_disjoint_summary.csv"
    mt_csv = RESULTS_DIR / "fewshot_multi_trial_summary.csv"
    if disjoint_csv.exists():
        df = pd.read_csv(disjoint_csv)
    elif mt_csv.exists():
        df = pd.read_csv(mt_csv)
    else:
        df = pd.read_csv(RESULTS_DIR / "fewshot_transfer_summary.csv")

    plt.rcParams.update({
        "font.size":       12,
        "axes.titlesize":  13,
        "axes.labelsize":  13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    K_values = sorted(df["K"].unique())
    # Categorical x-axis: evenly spaced positions, ticks labeled with K values.
    x_pos = np.arange(len(K_values))

    means = {}
    for model in ["hubert", "wavlm", "emotion2vec"]:
        sub = df[df["model"] == model]
        mean_per_K = sub.groupby("K")["mean_acc"].mean().reindex(K_values).to_numpy()
        std_per_K  = sub.groupby("K")["mean_acc"].std().reindex(K_values).to_numpy()
        means[model] = mean_per_K
        ax.plot(x_pos, mean_per_K, "o-",
                color=MODEL_COLORS[model], linewidth=2.5, markersize=8,
                label=MODEL_LABEL[model], zorder=3)
        ax.fill_between(x_pos,
                         mean_per_K - std_per_K,
                         mean_per_K + std_per_K,
                         color=MODEL_COLORS[model], alpha=0.18, zorder=2,
                         linewidth=0)

    # Chance baseline
    ax.axhline(y=0.25, color="#666666", linestyle="--", linewidth=1.4,
                label="4-class chance (0.25)", zorder=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel("K  (target-language labeled samples)")
    ax.set_ylabel("Accuracy (mean across 12 pairs)")
    ax.set_ylim(0.18, 0.92)
    ax.set_xlim(-0.35, len(K_values) - 1 + 0.55)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)

    # Endpoint value annotations
    for model in ["hubert", "wavlm", "emotion2vec"]:
        for idx, K in enumerate(K_values):
            if K not in (0, 100):
                continue
            v = means[model][idx]
            dx = -16 if K == 0 else 8
            dy = 8 if model != "emotion2vec" else -16
            ax.annotate(f"{v:.2f}",
                         xy=(idx, v), xytext=(dx, dy),
                         textcoords="offset points",
                         color=MODEL_COLORS[model], fontsize=11,
                         fontweight="bold",
                         ha="left" if K == 100 else "right")

    # Gap (Delta) annotations between best general SSL (HuBERT) and emotion2vec
    # at K=0 and K=100. Vertical bracket with label.
    def draw_gap_bracket(idx, K, ax_off, label_side):
        hub = means["hubert"][idx]
        e2v = means["emotion2vec"][idx]
        delta_pts = (hub - e2v) * 100
        x = idx + ax_off
        # vertical line spanning the gap
        ax.annotate("", xy=(x, hub), xytext=(x, e2v),
                     arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6))
        # label to the side
        label_x = x + 0.16 * label_side
        label_y = (hub + e2v) / 2
        ax.text(label_x, label_y,
                 f"$\\Delta \\approx {delta_pts:.0f}$ pt",
                 fontsize=12, fontweight="bold",
                 ha="left" if label_side == 1 else "right",
                 va="center", color="#222",
                 bbox=dict(boxstyle="round,pad=0.18", fc="white",
                            ec="#bbb", lw=0.8))

    # K=0 bracket: place at idx+0.45 with label to the right (avoids the
    # leftmost endpoint annotations).
    draw_gap_bracket(idx=0, K=0, ax_off=0.50, label_side=1)
    # K=100 bracket: place at idx-0.32 with label to the left.
    draw_gap_bracket(idx=len(K_values) - 1, K=100, ax_off=-0.40, label_side=-1)

    fig.tight_layout()
    out_png = RESULTS_DIR / "fewshot_transfer_clean.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    out_pdf = RESULTS_DIR / "fewshot_transfer_clean.pdf"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.18)
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
