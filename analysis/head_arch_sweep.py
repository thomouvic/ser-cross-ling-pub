"""Head-architecture sweep on Claim 1 speaker-independent.

The emotion2vec downstream classifier uses a two-layer head (hidden=256, ReLU);
our SuperbProbe uses a single linear layer by default. Sweep hidden in
{0 (linear), 256 (emotion2vec's head)} to check that the head choice does not
drive the Claim 1 result.

Single seed (RNG_SEED) per (model, dataset, hidden), 3 datasets x 3 models x 2
heads = 18 trainings.
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


def load_full(dataset: str, model: str):
    suffix = MODEL_ALIAS[model]
    base = CACHE_DIR / f"{dataset}__{suffix}"
    return np.load(str(base) + ".npy"), pd.read_csv(str(base) + ".csv")


def speaker_split(meta, seed=RNG_SEED, train_frac=0.6):
    rng = np.random.default_rng(seed)
    speakers = sorted(meta["speaker_id"].unique()); rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train).to_numpy()
    return train_mask, ~train_mask


def main():
    rows = []
    out = RESULTS_DIR / "head_arch_sweep.csv"
    for d in FULL_DATASETS:
        for m in MODEL_ALIAS:
            print(f"\n=== {d} x {m} ===", flush=True)
            try:
                emb, meta = load_full(d, m)
            except FileNotFoundError as e:
                print(f"  MISSING: {e}", flush=True); continue
            if emb.ndim != 3:
                print(f"  bad shape {emb.shape}", flush=True); continue

            classes = sorted(meta["emotion"].unique())
            label_to_int = {e: i for i, e in enumerate(classes)}
            train_mask, test_mask = speaker_split(meta)
            Xtr, Xte = emb[train_mask], emb[test_mask]
            ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])
            yte = np.asarray([label_to_int[y] for y in meta.loc[test_mask, "emotion"]])

            for hidden in [0, 256]:
                tag = "1-linear" if hidden == 0 else "2-linear-h256-relu"
                probe = SuperbProbe(n_layers=emb.shape[1], dim=emb.shape[2],
                                   n_classes=len(classes), hidden=hidden,
                                   lr=1e-3, weight_decay=1e-4, epochs=200,
                                   batch_size=128, verbose=False, seed=RNG_SEED)
                probe.fit(Xtr, ytr)
                pred = probe.predict(Xte)
                acc = float(accuracy_score(yte, pred))
                f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
                rows.append({
                    "dataset": d, "model": m, "hidden": hidden, "head": tag,
                    "n_classes": len(classes), "n_test": int(test_mask.sum()),
                    "accuracy": acc, "macro_f1": f1,
                })
                print(f"  head={tag:18s}  acc={acc:.4f}  f1={f1:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    print(f"\nSaved {out}", flush=True)
    # Pivot: rows=(model, dataset), cols=hidden, values=accuracy
    pivot = df.pivot_table(index=["model", "dataset"], columns="hidden",
                            values="accuracy")
    pivot["delta"] = pivot[256] - pivot[0]
    print("\n=== Accuracy by head architecture (hidden=0 vs 256) ===", flush=True)
    print(pivot.to_string(), flush=True)


if __name__ == "__main__":
    main()
