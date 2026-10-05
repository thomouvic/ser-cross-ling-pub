"""Cross-lingual RSA on emotion centroids using SUPERB-protocol embeddings.

Design: train ONE global SUPERB probe per model on the pooled multilingual data.
Use that probe's learned weighted sum as the canonical representation. Compute
per-(language, emotion) centroids in that single representation space and run
RSA across language pairs.

This is "option 2" from our methodology discussion: a single global representation
ensures all languages live in the same space for fair structural comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_embeddings, DATASETS, MODEL_ALIAS, DATASET_LANG
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
LANGS = ["en", "de", "zh", "bn"]
RNG_SEED = 20260505


def load_pooled(model: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate per-layer embeddings + metadata across all 5 datasets."""
    embs, metas = [], []
    for d in DATASETS:
        e, m = load_embeddings(d, model, layer=None)
        if e.ndim != 3:
            raise ValueError(f"Need per-layer for SUPERB; {d} {model} has shape {e.shape}")
        embs.append(e)
        metas.append(m)
    return np.concatenate(embs, axis=0), pd.concat(metas, ignore_index=True)


def fit_global_probe(emb: np.ndarray, meta: pd.DataFrame) -> SuperbProbe:
    """Train SuperbProbe on all data (pooled across languages) using
    speaker-independent split."""
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    rng = np.random.default_rng(RNG_SEED)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(0.8 * len(speakers))))  # 80% train, 20% val for probe selection
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()

    Xtr = emb[train_mask]
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])

    probe = SuperbProbe(n_layers=emb.shape[1], dim=emb.shape[2],
                       n_classes=len(EMOTION_4CLASS),
                       lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=256,
                       verbose=False, seed=RNG_SEED)
    probe.fit(Xtr, ytr)
    return probe


def centroid_distance_matrix(emb_2d: np.ndarray, meta: pd.DataFrame, language: str
                             ) -> np.ndarray:
    """4x4 cosine-distance matrix between emotion centroids in this language."""
    K = len(EMOTION_4CLASS)
    M = np.zeros((K, K))
    centroids = []
    for emo in EMOTION_4CLASS:
        mask = ((meta["emotion"] == emo) & (meta["language"] == language)).to_numpy()
        centroids.append(emb_2d[mask].mean(axis=0))
    centroids = np.stack(centroids, axis=0)
    for i in range(K):
        for j in range(K):
            M[i, j] = cosine(centroids[i], centroids[j])
    return M


def upper_triangle(M: np.ndarray) -> np.ndarray:
    K = M.shape[0]
    iu = np.triu_indices(K, k=1)
    return M[iu]


def plot_rsa_matrix(rsa: dict, model: str, out_path: Path):
    K = len(LANGS)
    M = np.full((K, K), np.nan)
    for i, la in enumerate(LANGS):
        for j, lb in enumerate(LANGS):
            if la == lb:
                M[i, j] = 1.0
            elif f"{la}-{lb}" in rsa:
                M[i, j] = rsa[f"{la}-{lb}"]["rho"]
            elif f"{lb}-{la}" in rsa:
                M[i, j] = rsa[f"{lb}-{la}"]["rho"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(K), LANGS)
    ax.set_yticks(range(K), LANGS)
    ax.set_title(f"RSA (Spearman, SUPERB) — {model}")
    for i in range(K):
        for j in range(K):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    rows = []
    for model in MODEL_ALIAS:
        emb, meta = load_pooled(model)
        meta["language"] = meta["dataset"].map(DATASET_LANG)

        probe = fit_global_probe(emb, meta)
        # Project all data through learned weighted sum
        emb_2d = probe.weighted_sum(emb)  # (N, D)

        rsa = {}
        for la, lb in combinations(LANGS, 2):
            Ma = centroid_distance_matrix(emb_2d, meta, la)
            Mb = centroid_distance_matrix(emb_2d, meta, lb)
            rho, p = spearmanr(upper_triangle(Ma), upper_triangle(Mb))
            rsa[f"{la}-{lb}"] = {"rho": rho, "p": p}

        for pair, vals in rsa.items():
            rows.append({"model": model, "pair": pair,
                         "spearman_rho": vals["rho"], "p": vals["p"]})
            print(f"{model:11s}  {pair}  rho={vals['rho']: .3f}  p={vals['p']:.3g}")

        plot_rsa_matrix(rsa, model, RESULTS_DIR / f"rsa_superb_{model}.png")

        mean_rho = np.mean([rsa[k]["rho"] for k in rsa])
        print(f"  mean rho = {mean_rho: .3f}")
        print()

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "rsa_superb_summary.csv"
    df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
