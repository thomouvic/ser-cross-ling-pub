"""Claim 1: matched re-implementation of the emotion2vec paper's 10-fold CV protocol.

The emotion2vec paper (Ma et al. 2024 ACL Findings) Section 4.3 describes its
language-generalization protocol (Tables 3 & 4):

    "random leave-one-out 10-fold CV ... at each fold, all samples within the
    dataset are randomly split into 80%, 10%, and 10% samples in training,
    validation, and testing sets."

This split is utterance-level, not speaker-grouped, so it differs from the
speaker-independent 60/40 split used elsewhere in the paper. To compare with the
published per-dataset numbers on matched terms, this script re-implements that
random 10-fold protocol with our probe.

Datasets: emodb_full (7 classes), ravdess_full (8 classes), subesco_full (7 classes).
Models: HuBERT-base, WavLM-base, emotion2vec_base. Note Table 4 does NOT include
HuBERT-base — head-to-head published comparison is WavLM-base vs emotion2vec.

Per-fold: train SUPERB weighted-sum probe on 80% (val unused; we don't do
val-based early stopping in our probe — fixed 200 epochs), test on 10%.
Aggregate mean and std across 10 folds.
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
N_FOLDS = 10

FULL_DATASETS = ["emodb_full", "ravdess_full", "subesco_full"]


def load_full(dataset: str, model: str) -> tuple[np.ndarray, pd.DataFrame]:
    suffix = MODEL_ALIAS[model]
    base = CACHE_DIR / f"{dataset}__{suffix}"
    emb = np.load(str(base) + ".npy")
    meta = pd.read_csv(str(base) + ".csv")
    return emb, meta


def random_10fold_indices(n: int, seed: int = RNG_SEED) -> list[np.ndarray]:
    """Return a list of 10 disjoint test-index arrays (one per fold).
    The union covers all of [0, n)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, N_FOLDS)
    return [np.sort(f) for f in folds]


def probe_one_fold(emb: np.ndarray, y: np.ndarray, n_classes: int,
                    test_idx: np.ndarray, fold_seed: int) -> tuple[float, float]:
    n = emb.shape[0]
    is_test = np.zeros(n, dtype=bool)
    is_test[test_idx] = True
    Xtr, Xte = emb[~is_test], emb[is_test]
    ytr, yte = y[~is_test], y[is_test]

    probe = SuperbProbe(n_layers=emb.shape[1], dim=emb.shape[2],
                       n_classes=n_classes, lr=1e-3, weight_decay=1e-4,
                       epochs=200, batch_size=128, seed=fold_seed)
    probe.fit(Xtr, ytr)
    pred = probe.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    return acc, f1


def cv10_for(dataset: str, model: str) -> dict:
    emb, meta = load_full(dataset, model)
    if emb.ndim != 3:
        raise ValueError(f"Need per-layer (N, L, D); got {emb.shape}")

    classes = sorted(meta["emotion"].unique())
    label_to_int = {e: i for i, e in enumerate(classes)}
    y = np.asarray([label_to_int[e] for e in meta["emotion"]])

    fold_test_indices = random_10fold_indices(len(meta), seed=RNG_SEED)
    fold_acc, fold_f1 = [], []
    for i, test_idx in enumerate(fold_test_indices):
        acc, f1 = probe_one_fold(emb, y, len(classes), test_idx,
                                  fold_seed=RNG_SEED + i)
        fold_acc.append(acc); fold_f1.append(f1)
        print(f"  [{dataset} x {model}] fold {i+1}/10: "
              f"acc={acc:.4f}  f1={f1:.4f}", flush=True)

    return {
        "dataset": dataset, "model": model,
        "n_classes": len(classes), "classes": ",".join(classes),
        "n_total": len(meta),
        "acc_mean": float(np.mean(fold_acc)),
        "acc_std":  float(np.std(fold_acc, ddof=1)),
        "acc_per_fold": ";".join(f"{a:.4f}" for a in fold_acc),
        "f1_mean":  float(np.mean(fold_f1)),
        "f1_std":   float(np.std(fold_f1, ddof=1)),
        "f1_per_fold": ";".join(f"{f:.4f}" for f in fold_f1),
    }


def main():
    out = RESULTS_DIR / "probe_superb_full_cv10_summary.csv"
    # Resume support: load any prior rows and skip combos already done.
    if out.exists():
        prior = pd.read_csv(out)
        rows = prior.to_dict(orient="records")
        done = {(r["dataset"], r["model"]) for r in rows}
        print(f"[resume] {len(rows)} rows already in CSV, skipping: {sorted(done)}",
              flush=True)
    else:
        rows = []
        done = set()

    for d in FULL_DATASETS:
        for m in MODEL_ALIAS:
            if (d, m) in done:
                continue
            print(f"\n=== {d} x {m} ===", flush=True)
            try:
                res = cv10_for(d, m)
                rows.append(res)
                print(f"  SUMMARY {d} x {m}: "
                      f"acc {res['acc_mean']*100:.2f} +/- {res['acc_std']*100:.2f}  "
                      f"f1 {res['f1_mean']*100:.2f} +/- {res['f1_std']*100:.2f}",
                      flush=True)
            except FileNotFoundError as e:
                print(f"  MISSING: {e}", flush=True)
            # Incremental save so a timeout doesn't lose all results.
            pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\nFinal results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
