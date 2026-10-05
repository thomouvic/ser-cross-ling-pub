"""SUPERB-style probing on the full emotion sets of three corpora (Claim 1).

Speaker-independent probing with our frozen SUPERB-style weighted-sum probe on
the full per-corpus label sets used in the emotion2vec paper's per-dataset results.

Datasets: emodb_full (7 classes), ravdess_full (8 classes), subesco_full (7 classes).
Models: HuBERT, WavLM, emotion2vec_base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import CACHE_DIR, MODEL_ALIAS
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260505

FULL_DATASETS = ["emodb_full", "ravdess_full", "subesco_full"]


def load_full(dataset: str, model: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load per-layer embeddings + metadata for a *_full dataset variant.
    These are produced by extraction with the full-emotion-set loaders."""
    suffix = MODEL_ALIAS[model]
    base = CACHE_DIR / f"{dataset}__{suffix}"
    emb = np.load(str(base) + ".npy")
    meta = pd.read_csv(str(base) + ".csv")
    return emb, meta


def speaker_split(meta: pd.DataFrame, train_frac: float = 0.6,
                  rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()
    return train_mask, ~train_mask


def bootstrap_metric(y_true, y_pred, fn, n=1000, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    point = float(fn(y_true, y_pred))
    if len(y_true) < 2:
        return point, point, point
    samples = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        samples.append(fn(y_true[idx], y_pred[idx]))
    arr = np.asarray(samples)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def probe_full(dataset: str, model: str) -> dict:
    emb, meta = load_full(dataset, model)
    if emb.ndim != 3:
        raise ValueError(f"Need per-layer (N, L, D); got {emb.shape}")

    train_mask, test_mask = speaker_split(meta)
    Xtr, Xte = emb[train_mask], emb[test_mask]

    # Use this dataset's actual emotion classes (not the 4-class intersection)
    classes = sorted(meta["emotion"].unique())
    label_to_int = {e: i for i, e in enumerate(classes)}
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])
    yte = np.asarray([label_to_int[y] for y in meta.loc[test_mask, "emotion"]])

    probe = SuperbProbe(n_layers=emb.shape[1], dim=emb.shape[2],
                       n_classes=len(classes), lr=1e-3, weight_decay=1e-4,
                       epochs=200, batch_size=128, seed=RNG_SEED)
    probe.fit(Xtr, ytr)
    pred = probe.predict(Xte)

    acc, acc_lo, acc_hi = bootstrap_metric(yte, pred, accuracy_score)
    f1, f1_lo, f1_hi = bootstrap_metric(
        yte, pred, lambda y, p: f1_score(y, p, average="macro", zero_division=0))

    return {
        "dataset": dataset, "model": model,
        "n_classes": len(classes), "classes": ",".join(classes),
        "n_test": int(test_mask.sum()),
        "n_test_speakers": meta.loc[test_mask, "speaker_id"].nunique(),
        "accuracy": acc, "accuracy_ci_lo": acc_lo, "accuracy_ci_hi": acc_hi,
        "macro_f1": f1, "macro_f1_ci_lo": f1_lo, "macro_f1_ci_hi": f1_hi,
    }


def main():
    rows = []
    for d in FULL_DATASETS:
        for m in MODEL_ALIAS:
            try:
                res = probe_full(d, m)
                rows.append(res)
                print(f"{d:14s} x {m:11s}  n_classes={res['n_classes']}  "
                      f"acc={res['accuracy']:.3f} [{res['accuracy_ci_lo']:.3f}, {res['accuracy_ci_hi']:.3f}]  "
                      f"F1={res['macro_f1']:.3f}")
            except FileNotFoundError as e:
                print(f"{d:14s} x {m:11s}  MISSING: {e}")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "probe_superb_full_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
