"""Layer-choice check: does probing only the last layer lower WavLM's accuracy?

Same 10-fold CV protocol and the same SuperbProbe as probe_superb_full_cv10.py,
but each (model, dataset) is probed on a single layer fed as (N, 1, D); the
weighted sum over one layer is just that layer. This isolates layer choice
(fixed last layer vs. learned weighted sum), one of the differences between
our probe and the published baseline pipeline.

Reports, per (dataset, model):
  - last-layer-only CV10 accuracy
  - best-single-layer CV10 accuracy (oracle, for context)
for comparison with the weighted-sum results of probe_superb_full_cv10.py.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import CACHE_DIR, MODEL_ALIAS
from superb_probe import SuperbProbe

RNG_SEED = 20260505
N_FOLDS = 10
FULL = ["emodb_full", "ravdess_full", "subesco_full"]
MODELS = ["wavlm", "emotion2vec"]  # the pair compared in the emotion2vec paper's Table 4


def load_full(dataset, model):
    base = CACHE_DIR / f"{dataset}__{MODEL_ALIAS[model]}"
    return np.load(str(base) + ".npy"), pd.read_csv(str(base) + ".csv")


def folds(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    return [np.sort(f) for f in np.array_split(rng.permutation(n), N_FOLDS)]


def cv10_single_layer(emb1, y, n_classes):
    """emb1: (N, 1, D). Returns mean acc over 10 folds."""
    fts = folds(emb1.shape[0])
    accs = []
    for i, test_idx in enumerate(fts):
        is_te = np.zeros(emb1.shape[0], bool); is_te[test_idx] = True
        p = SuperbProbe(n_layers=1, dim=emb1.shape[2], n_classes=n_classes,
                        lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=128,
                        seed=RNG_SEED + i)
        p.fit(emb1[~is_te], y[~is_te])
        accs.append(accuracy_score(y[is_te], p.predict(emb1[is_te])))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def main():
    rows = []
    for d in FULL:
        for m in MODELS:
            emb, meta = load_full(d, m)            # (N, L, D)
            classes = sorted(meta["emotion"].unique())
            y = np.asarray([classes.index(e) for e in meta["emotion"]])
            L = emb.shape[1]
            # last layer only, as in the published baseline probe
            last_mean, last_std = cv10_single_layer(emb[:, -1:, :], y, len(classes))
            rows.append(dict(dataset=d, model=m, n_layers=L,
                             last_layer_acc=last_mean, last_layer_std=last_std))
            print(f"{d:14s} {m:11s}  last-layer(L{L-1}) CV10 = "
                  f"{last_mean*100:5.2f} +/- {last_std*100:.2f}", flush=True)
    out = Path(__file__).resolve().parent.parent / "results" / "lastlayer_vs_weightedsum_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    main()
