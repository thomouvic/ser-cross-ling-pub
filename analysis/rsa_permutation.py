"""Per-utterance permutation tests for RSA significance (RQ2).

Each language pair's RSA correlation is computed over only 6 entries (the upper
triangle of a 4x4 emotion-centroid distance matrix), so standard inference under
that small N is unreliable. A permutation null avoids that problem.

Permutation null:
  For each (model, language A, language B) pair:
    Compute observed Spearman rho on (UT(D_A), UT(D_B)).
    For p in 1..N_PERMS:
      Shuffle utterance-level emotion labels WITHIN language A (preserve emo
      distribution; randomize assignment). Recompute centroids -> D_A_perm.
      Compute rho_p on (UT(D_A_perm), UT(D_B)).
    p-value (one-sided, expecting positive rho) = fraction of rho_p >= rho_obs.

Why utterance-level rather than emotion-label permutation? With only 4 emotions
there are 4! = 24 permutations of the centroid-row assignment, which gives a
discrete null distribution with limited resolution. Permuting at utterance
level keeps the marginal emotion distribution but breaks the *within-utterance*
structure, providing a proper continuous null.

This script reuses rsa_superb's pooling + global probe to compute the
SUPERB-pooled representation, then does the permutation test on top.
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


def centroid_dmat(emb_2d, emos_in_lang) -> np.ndarray:
    """4x4 cosine-distance matrix from emotion centroids in a language.
    `emos_in_lang` is per-utterance emotion label array for THAT language."""
    K = len(EMOTION_4CLASS)
    cents = np.stack([emb_2d[emos_in_lang == e].mean(axis=0) for e in EMOTION_4CLASS])
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            M[i, j] = cosine(cents[i], cents[j])
    return M


def upper_triangle(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


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

        # Per-language emotion arrays + projected embeddings (so we can fast-shuffle)
        lang_data = {}
        for la in LANGS:
            mask = (meta["language"] == la).to_numpy()
            lang_data[la] = {
                "emb": emb_2d[mask],
                "emo": meta.loc[mask, "emotion"].to_numpy(),
            }

        for la, lb in combinations(LANGS, 2):
            Ea = np.asarray([e for e in lang_data[la]["emo"]])
            Eb = np.asarray([e for e in lang_data[lb]["emo"]])
            Da_obs = centroid_dmat(lang_data[la]["emb"], Ea)
            Db = centroid_dmat(lang_data[lb]["emb"], Eb)
            ut_b = upper_triangle(Db)
            rho_obs, _ = spearmanr(upper_triangle(Da_obs), ut_b)

            # Permutation null on language A
            null_rhos = np.empty(N_PERMS, dtype=float)
            for p in range(N_PERMS):
                Ea_perm = rng_master.permutation(Ea)
                Da_perm = centroid_dmat(lang_data[la]["emb"], Ea_perm)
                rho_p, _ = spearmanr(upper_triangle(Da_perm), ut_b)
                null_rhos[p] = rho_p

            # One-sided p-value (expecting positive RSA): proportion >= observed
            p_value = float(np.mean(null_rhos >= rho_obs))
            null_mean = float(null_rhos.mean())
            null_std  = float(null_rhos.std(ddof=1))

            rows.append({
                "model": model, "pair": f"{la}-{lb}",
                "rho_observed": float(rho_obs),
                "p_value": p_value,
                "null_mean": null_mean, "null_std": null_std,
                "n_perms": N_PERMS,
                "n_utts_a": int(len(Ea)), "n_utts_b": int(len(Eb)),
            })
            print(f"  {la}-{lb}  rho={rho_obs:+.3f}  null={null_mean:+.3f} ± {null_std:.3f}  "
                  f"p={p_value:.3g}", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "rsa_permutation_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}", flush=True)
    print("\nSignificance summary (alpha=0.05):", flush=True)
    sig = df.assign(significant=df["p_value"] < 0.05)
    print(sig[["model", "pair", "rho_observed", "p_value", "significant"]]
            .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
