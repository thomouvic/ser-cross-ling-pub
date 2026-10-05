"""Layer-weight visualization (free side-finding for the paper).

Plot the SUPERB-probe learned alphas (softmax over hidden states) per
(model, dataset). Two panels:
  - Heatmap: dataset (rows) × layer index (cols), per model.
  - Optional line plot per model showing alphas overlaid across datasets.

Tells the story of which layers each model relies on for emotion in each
language, and whether different languages prefer different layers.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    df = pd.read_csv(RESULTS_DIR / "probe_superb_layer_weights.csv")

    models = sorted(df["model"].unique())
    datasets = sorted(df["dataset"].unique())

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
    })

    fig, axes = plt.subplots(1, len(models), figsize=(6.0 * len(models), 4.0),
                              sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        n_layers = sub["layer"].max() + 1

        # Heatmap matrix: rows=datasets, cols=layers
        mat = np.zeros((len(datasets), n_layers))
        for i, d in enumerate(datasets):
            row = sub[sub["dataset"] == d].sort_values("layer")
            mat[i, :len(row)] = row["alpha"].to_numpy()

        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=df["alpha"].min(), vmax=df["alpha"].max())
        ax.set_title(f"{model}  (L={n_layers})")
        ax.set_xlabel("layer (0 = input embedding)")
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)

    cbar = fig.colorbar(im, ax=axes, label="alpha (softmax weight)",
                         shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    out_png = RESULTS_DIR / "layer_weights_heatmap.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"Saved {out_png}")

    # Line plot: alphas across layers, one curve per dataset, faceted by model.
    fig2, axes2 = plt.subplots(1, len(models), figsize=(5 * len(models), 3.2),
                                 sharey=True)
    if len(models) == 1:
        axes2 = [axes2]
    cmap = plt.get_cmap("tab10")
    for ax, model in zip(axes2, models):
        sub = df[df["model"] == model]
        n_layers = sub["layer"].max() + 1
        for i, d in enumerate(datasets):
            row = sub[sub["dataset"] == d].sort_values("layer")
            ax.plot(row["layer"], row["alpha"], marker="o", color=cmap(i),
                    label=d)
        ax.set_title(f"{model} (L={n_layers})")
        ax.set_xlabel("layer index")
        ax.grid(True, alpha=0.3)
    axes2[0].set_ylabel("alpha")
    axes2[-1].legend(loc="upper right", fontsize=8)
    fig2.suptitle("SUPERB layer-weight profile across datasets")
    out_png2 = RESULTS_DIR / "layer_weights_lines.png"
    plt.savefig(out_png2, dpi=160, bbox_inches="tight")
    print(f"Saved {out_png2}")

    # Per-model: which layer has highest alpha per dataset
    print("\n=== Argmax layer per (model, dataset) ===")
    summary = (df.loc[df.groupby(["model", "dataset"])["alpha"].idxmax(),
                       ["model", "dataset", "layer", "alpha"]]
                 .sort_values(["model", "dataset"])
                 .reset_index(drop=True))
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "layer_weights_argmax.csv", index=False)


if __name__ == "__main__":
    main()
