"""Clean paper figure for SUPERB layer weights.

Two-panel full-width figure:
  Left panel: line plot of mean alpha (across datasets) vs *normalized* layer
              depth (0 = input embedding, 1 = last transformer), one curve per
              model with shaded +/-1 std band. Conveys the headline that
              emotion2vec is high at depth 0 then decays, while HuBERT and
              WavLM peak mid-stack.
  Right panel: 3 stacked heatmaps (one per model) of alpha values per
               (dataset, layer). Per-dataset detail for readers who want it.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_COLORS = {
    "hubert":      "#1f4e79",
    "wavlm":       "#2e7d6b",
    "emotion2vec": "#c64241",
}
MODEL_LABEL = {
    "hubert":      "HuBERT (L=12)",
    "wavlm":       "WavLM (L=12)",
    "emotion2vec": "emotion2vec_base (L=8)",
}


def main():
    df = pd.read_csv(RESULTS_DIR / "probe_superb_layer_weights.csv")

    plt.rcParams.update({
        "font.size":       11,
        "axes.titlesize":  12,
        "axes.labelsize":  11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig = plt.figure(figsize=(13.5, 4.6))
    outer = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.25,
                                figure=fig)

    # -------- LEFT PANEL: line plot, mean alpha vs normalized depth --------
    ax_line = fig.add_subplot(outer[0])
    for model in ["hubert", "wavlm", "emotion2vec"]:
        sub = df[df["model"] == model]
        n_layers = sub["layer"].max() + 1   # includes input embedding
        # alpha[dataset, layer]
        datasets = sorted(sub["dataset"].unique())
        mat = np.zeros((len(datasets), n_layers))
        for i, d in enumerate(datasets):
            row = sub[sub["dataset"] == d].sort_values("layer")
            mat[i, :] = row["alpha"].to_numpy()
        # normalized depth: layer / (n_layers - 1)
        depth = np.arange(n_layers) / (n_layers - 1)
        mean_alpha = mat.mean(axis=0)
        std_alpha  = mat.std(axis=0, ddof=1)
        ax_line.plot(depth, mean_alpha, "o-",
                     color=MODEL_COLORS[model], linewidth=2.4, markersize=6,
                     label=MODEL_LABEL[model])
        ax_line.fill_between(depth, mean_alpha - std_alpha, mean_alpha + std_alpha,
                              color=MODEL_COLORS[model], alpha=0.18, linewidth=0)
    ax_line.set_xlabel("Normalized layer depth (0 = input embedding, 1 = last transformer)")
    ax_line.set_ylabel("Mean SUPERB layer weight $\\alpha$ (across 5 datasets)")
    ax_line.set_xlim(-0.03, 1.03)
    ax_line.set_ylim(0, max(0.42, df["alpha"].max() * 1.05))
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(loc="upper right", framealpha=0.95)
    ax_line.set_title("(a) Mean layer weight by model")

    # -------- RIGHT PANEL: 3 stacked heatmaps, one per model --------
    inner = gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=outer[1],
        width_ratios=[1.0, 0.04], hspace=0.55, wspace=0.06,
    )
    vmin, vmax = df["alpha"].min(), df["alpha"].max()
    datasets_sorted = sorted(df["dataset"].unique())
    im = None
    model_order = ["hubert", "wavlm", "emotion2vec"]
    for row_i, model in enumerate(model_order):
        ax = fig.add_subplot(inner[row_i, 0])
        sub = df[df["model"] == model]
        n_layers = sub["layer"].max() + 1
        mat = np.zeros((len(datasets_sorted), n_layers))
        for i, d in enumerate(datasets_sorted):
            r = sub[sub["dataset"] == d].sort_values("layer")
            mat[i, :] = r["alpha"].to_numpy()
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=9)
        ax.set_yticks(range(len(datasets_sorted)))
        ax.set_yticklabels(datasets_sorted, fontsize=9)
        ax.set_title(MODEL_LABEL[model], fontsize=11)
        if row_i == 2:
            ax.set_xlabel("layer index (0 = input embedding)")

    # Colorbar in the rightmost narrow column, spanning all three rows
    cax = fig.add_subplot(inner[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("$\\alpha$ (softmax weight)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.text(0.59, 0.97, "(b) Per-(model, dataset) layer weights",
              fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = RESULTS_DIR / "layer_weights_paperfig.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
