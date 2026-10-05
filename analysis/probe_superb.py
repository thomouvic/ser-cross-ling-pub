"""SUPERB-protocol linear probing.

For each (model, dataset) pair:
  - Load per-layer embeddings (N, L, D).
  - Speaker-independent split.
  - Train a SuperbProbe (learnable layer weights + linear head, frozen encoder).
  - Bootstrap CIs on the test set.

This replaces the layer-fixed probe.py for the main RQ1 numbers and is the
standard protocol used by the emotion2vec paper and SUPERB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_embeddings, DATASETS, MODEL_ALIAS
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
RNG_SEED = 20260505


def speaker_split(meta: pd.DataFrame, train_frac: float = 0.6,
                  rng=None) -> tuple[np.ndarray, np.ndarray]:
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
    n_test = len(y_true)
    if n_test < 2:
        return point, point, point
    samples = []
    for _ in range(n):
        idx = rng.integers(0, n_test, size=n_test)
        samples.append(fn(y_true[idx], y_pred[idx]))
    arr = np.asarray(samples)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def probe_one(dataset: str, model: str) -> dict:
    emb, meta = load_embeddings(dataset, model, layer=None)
    if emb.ndim != 3:
        raise ValueError(f"Expected per-layer (N, L, D), got {emb.shape}")

    train_mask, test_mask = speaker_split(meta)
    Xtr, Xte = emb[train_mask], emb[test_mask]
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])
    yte = np.asarray([label_to_int[y] for y in meta.loc[test_mask, "emotion"]])

    n_layers = emb.shape[1]
    dim = emb.shape[2]

    probe = SuperbProbe(n_layers=n_layers, dim=dim, n_classes=len(EMOTION_4CLASS),
                        lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=128,
                        verbose=False, seed=RNG_SEED)
    probe.fit(Xtr, ytr)
    pred = probe.predict(Xte)

    acc, acc_lo, acc_hi = bootstrap_metric(yte, pred, accuracy_score)
    f1, f1_lo, f1_hi = bootstrap_metric(
        yte, pred, lambda y, p: f1_score(y, p, average="macro", zero_division=0),
    )
    weights = probe.layer_weights().tolist()

    return {
        "dataset": dataset,
        "model": model,
        "n_layers": n_layers,
        "n_test": int(test_mask.sum()),
        "n_test_speakers": meta.loc[test_mask, "speaker_id"].nunique(),
        "accuracy": acc,
        "accuracy_ci_lo": acc_lo,
        "accuracy_ci_hi": acc_hi,
        "macro_f1": f1,
        "macro_f1_ci_lo": f1_lo,
        "macro_f1_ci_hi": f1_hi,
        "layer_weights": weights,
    }


def main():
    rows = []
    for d in DATASETS:
        for m in MODEL_ALIAS:
            res = probe_one(d, m)
            rows.append(res)
            print(f"{d:8s} x {m:11s}  acc={res['accuracy']:.3f} [{res['accuracy_ci_lo']:.3f}, {res['accuracy_ci_hi']:.3f}]  "
                  f"F1={res['macro_f1']:.3f} [{res['macro_f1_ci_lo']:.3f}, {res['macro_f1_ci_hi']:.3f}]")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "probe_superb_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Save layer-weight matrix for inspection
    weights_rows = []
    for r in rows:
        for layer_idx, w in enumerate(r["layer_weights"]):
            weights_rows.append({"dataset": r["dataset"], "model": r["model"], "layer": layer_idx, "alpha": w})
    pd.DataFrame(weights_rows).to_csv(RESULTS_DIR / "probe_superb_layer_weights.csv", index=False)
    print(f"Saved {RESULTS_DIR}/probe_superb_layer_weights.csv")


if __name__ == "__main__":
    main()
