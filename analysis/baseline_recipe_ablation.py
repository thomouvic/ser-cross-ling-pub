"""Probe-recipe ablation: which configuration choices lower WavLM's probe accuracy?

The published WavLM-base baseline is much lower than the WavLM accuracy our
probe obtains under the same random 10-fold protocol. The last-layer check
(lastlayer_vs_weightedsum_check.py) shows that layer choice alone accounts for
only a small part of that difference. This script tests two further recipe
differences: (1) no feature standardization, and (2) a short, low-learning-rate
training budget (lr 1e-4, about 20 epochs, as in the emotion2vec/EmoBox
downstream recipe).

Standardization is a plausible WavLM-specific factor: WavLM/wav2vec2 features
contain a few large-magnitude dimensions that can dominate an unstandardized
linear probe, whereas emotion2vec's distilled output is better conditioned.

Features: last layer. Datasets: EmoDB, RAVDESS. Models: WavLM, emotion2vec.
Protocol: random 10-fold CV.

Configs:
  C1 std=T  lr1e-3 200ep   our recipe (control)
  C2 std=F  lr1e-3 200ep   no standardization
  C3 std=T  lr1e-4  20ep   short training budget
  C4 std=F  lr1e-4  20ep   no standardization + short budget
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
DATASETS = ["emodb_full", "ravdess_full"]
MODELS = ["wavlm", "emotion2vec"]
CONFIGS = [
    dict(tag="C1_std_long",   standardize=True,  lr=1e-3, epochs=200),
    dict(tag="C2_nostd_long", standardize=False, lr=1e-3, epochs=200),
    dict(tag="C3_std_short",  standardize=True,  lr=1e-4, epochs=20),
    dict(tag="C4_nostd_short",standardize=False, lr=1e-4, epochs=20),
]


def load_full(dataset, model):
    base = CACHE_DIR / f"{dataset}__{MODEL_ALIAS[model]}"
    return np.load(str(base) + ".npy"), pd.read_csv(str(base) + ".csv")


def folds(n):
    rng = np.random.default_rng(RNG_SEED)
    return [np.sort(f) for f in np.array_split(rng.permutation(n), N_FOLDS)]


def cv10(emb1, y, n_classes, cfg):
    fts = folds(emb1.shape[0]); accs = []
    for i, test_idx in enumerate(fts):
        is_te = np.zeros(emb1.shape[0], bool); is_te[test_idx] = True
        p = SuperbProbe(n_layers=1, dim=emb1.shape[2], n_classes=n_classes,
                        lr=cfg["lr"], weight_decay=1e-4, epochs=cfg["epochs"],
                        batch_size=128, seed=RNG_SEED + i,
                        standardize=cfg["standardize"])
        p.fit(emb1[~is_te], y[~is_te])
        accs.append(accuracy_score(y[is_te], p.predict(emb1[is_te])))
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def main():
    rows = []
    for d in DATASETS:
        for m in MODELS:
            emb, meta = load_full(d, m)
            classes = sorted(meta["emotion"].unique())
            y = np.asarray([classes.index(e) for e in meta["emotion"]])
            last = emb[:, -1:, :]   # last layer, (N,1,D)
            for cfg in CONFIGS:
                mu, sd = cv10(last, y, len(classes), cfg)
                rows.append(dict(dataset=d, model=m, config=cfg["tag"],
                                 acc=mu, std=sd))
                print(f"{d:13s} {m:11s} {cfg['tag']:14s} = "
                      f"{mu*100:5.2f} +/- {sd*100:.2f}", flush=True)
            print(flush=True)
    out = Path(__file__).resolve().parent.parent / "results" / "baseline_recipe_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
