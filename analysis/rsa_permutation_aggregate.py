"""Aggregate permutation test for RSA across language pairs.

The per-pair test (rsa_permutation.py) is underpowered for individual pairs
because N=6 entries per upper triangle. Aggregate test asks the stronger
question: "is the MEAN rho across all 6 language pairs higher than chance?"

For each permutation: shuffle utterance-level emotion labels independently
within EVERY language, recompute all 6 pair rhos, take the mean. Compare observed
mean rho to that null distribution.

Under H0: no language preserves emotion structure that matches any other.
Under H1: at least some pairs show real alignment, lifting the mean.
"""

from __future__ import annotations

import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
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
N_PERMS = 1000


def load_pooled(model: str):
    embs, metas = [], []
    for d in DATASETS:
        e, m = load_embeddings(d, model, layer=None)
        embs.append(e); metas.append(m)
    return np.concatenate(embs, axis=0), pd.concat(metas, ignore_index=True)


def fit_global_probe(emb, meta):
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    rng = np.random.default_rng(RNG_SEED)
    speakers = sorted(meta["speaker_id"].unique()); rng.shuffle(speakers)
    n_train = max(1, int(round(0.8 * len(speakers))))
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


def centroid_dmat(emb_2d, emos):
    K = len(EMOTION_4CLASS)
    cents = np.stack([emb_2d[emos == e].mean(axis=0) for e in EMOTION_4CLASS])
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = cosine(cents[i], cents[j])
    return M


def upper_triangle(M):
    return M[np.triu_indices(M.shape[0], k=1)]


def all_pair_rhos(lang_data: dict, langs: list) -> list[float]:
    """Compute Spearman rho for every unordered language pair."""
    # Cache per-language UT(D)
    uts = {la: upper_triangle(centroid_dmat(lang_data[la]["emb"],
                                              lang_data[la]["emo"]))
            for la in langs}
    rhos = []
    for la, lb in combinations(langs, 2):
        rho, _ = spearmanr(uts[la], uts[lb])
        rhos.append(rho)
    return rhos


def main():
    rng_master = np.random.default_rng(RNG_SEED)
    rows = []

    for model in MODEL_ALIAS:
        print(f"\n=== {model} ===", flush=True)
        emb, meta = load_pooled(model)
        meta["language"] = meta["dataset"].map(DATASET_LANG)

        print("  fitting global SUPERB probe...", flush=True)
        probe = fit_global_probe(emb, meta)
        emb_2d = probe.weighted_sum(emb)

        lang_data = {}
        for la in LANGS:
            mask = (meta["language"] == la).to_numpy()
            lang_data[la] = {
                "emb": emb_2d[mask],
                "emo": meta.loc[mask, "emotion"].to_numpy(),
            }

        # Observed mean rho
        obs_rhos = all_pair_rhos(lang_data, LANGS)
        obs_mean = float(np.mean(obs_rhos))
        obs_median = float(np.median(obs_rhos))
        print(f"  observed: per-pair rhos = {[f'{r:+.3f}' for r in obs_rhos]}",
              flush=True)
        print(f"  observed: mean rho = {obs_mean:+.4f}, median = {obs_median:+.4f}",
              flush=True)

        # Permutation null: independently shuffle emotion labels in each language.
        null_means = np.empty(N_PERMS)
        null_medians = np.empty(N_PERMS)
        perm_lang_data = {la: {"emb": lang_data[la]["emb"], "emo": None}
                           for la in LANGS}
        for p in range(N_PERMS):
            for la in LANGS:
                perm_lang_data[la]["emo"] = rng_master.permutation(lang_data[la]["emo"])
            rhos_p = all_pair_rhos(perm_lang_data, LANGS)
            null_means[p] = np.mean(rhos_p)
            null_medians[p] = np.median(rhos_p)

        p_value_mean = float(np.mean(null_means >= obs_mean))
        p_value_median = float(np.mean(null_medians >= obs_median))
        print(f"  null:    mean rho = {null_means.mean():+.4f} ± {null_means.std(ddof=1):.4f}",
              flush=True)
        print(f"  AGGREGATE TEST (mean rho across 6 pairs):",
              flush=True)
        print(f"    observed = {obs_mean:+.4f},  null mean = {null_means.mean():+.4f},  "
              f"null std = {null_means.std(ddof=1):.4f}", flush=True)
        print(f"    p-value (one-sided) = {p_value_mean:.4g}", flush=True)
        print(f"  AGGREGATE TEST (median rho across 6 pairs):", flush=True)
        print(f"    observed = {obs_median:+.4f},  null mean = {null_medians.mean():+.4f}",
              flush=True)
        print(f"    p-value (one-sided) = {p_value_median:.4g}", flush=True)

        rows.append({
            "model": model,
            "obs_mean_rho": obs_mean,
            "obs_median_rho": obs_median,
            "null_mean": null_means.mean(),
            "null_std": null_means.std(ddof=1),
            "p_mean":   p_value_mean,
            "p_median": p_value_median,
            "n_perms":  N_PERMS,
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "rsa_permutation_aggregate.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}", flush=True)
    print("\n=== Final aggregate test summary ===", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
