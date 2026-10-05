"""Leave-one-speaker-out CV on the two smallest corpora (EmoDB, ESD-ZH, 10 speakers each).

A split-stability check for the two small per-language corpora. This runs
leave-one-speaker-out CV with the same SUPERB-style weighted-sum probe used for
the per-language results (RQ1): for each speaker, that speaker's utterances are
the test fold and the remaining nine speakers train. Reports mean +/- std of
4-class accuracy across the speaker folds.

Complements the 5-seed multi-split results (multi_seed_runner.py): LOGO is a
distinct, exhaustive partition (every speaker held out once).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

# Use all CPUs Slurm gave us for the torch matrix ops.
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_embeddings
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
SEED = 20260505
DATASETS = ["emodb", "esd_zh"]
MODELS = ["hubert", "wavlm", "emotion2vec"]


def logo_one(dataset: str, model: str) -> dict:
    emb, meta = load_embeddings(dataset, model)
    if emb.ndim != 3:
        raise ValueError(f"Expected per-layer (N, L, D), got {emb.shape}")
    # Restrict to the 4-class intersection (defensive; caches should already be 4-class).
    keep = meta["emotion"].isin(EMOTION_4CLASS).to_numpy()
    emb, meta = emb[keep], meta.loc[keep].reset_index(drop=True)
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    y = np.asarray([label_to_int[e] for e in meta["emotion"]])
    n_layers, dim = emb.shape[1], emb.shape[2]

    speakers = sorted(meta["speaker_id"].unique())
    fold_acc = []
    for s in speakers:
        test_mask = (meta["speaker_id"] == s).to_numpy()
        train_mask = ~test_mask
        probe = SuperbProbe(n_layers=n_layers, dim=dim, n_classes=len(EMOTION_4CLASS),
                            lr=1e-3, weight_decay=1e-4, epochs=200, batch_size=128,
                            verbose=False, seed=SEED)
        probe.fit(emb[train_mask], y[train_mask])
        pred = probe.predict(emb[test_mask])
        acc = float(accuracy_score(y[test_mask], pred))
        fold_acc.append(acc)
        print(f"  {dataset:7s} {model:11s} heldout_spk={str(s):>6s} "
              f"n_test={int(test_mask.sum()):5d} acc={acc:.3f}", flush=True)
    fa = np.asarray(fold_acc)
    return {
        "dataset": dataset, "model": model, "n_folds": len(speakers),
        "mean_acc": float(fa.mean()), "std_acc": float(fa.std(ddof=1)),
        "min_acc": float(fa.min()), "max_acc": float(fa.max()),
    }


def main():
    out = RESULTS_DIR / "logo_cv_summary.csv"
    # Resume: keep any (dataset, model) cells already computed in a prior run.
    rows = []
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        rows = prev.to_dict("records")
        done = {(r["dataset"], r["model"]) for r in rows}
        print(f"Resuming; {len(done)} cell(s) already done: {sorted(done)}", flush=True)
    for d in DATASETS:
        for m in MODELS:
            if (d, m) in done:
                continue
            r = logo_one(d, m)
            rows.append(r)
            print(f"== {d} {m}: mean={r['mean_acc']:.3f} std={r['std_acc']:.3f} "
                  f"range=[{r['min_acc']:.3f}, {r['max_acc']:.3f}]", flush=True)
            # Write incrementally so a wall-clock timeout still leaves partial results.
            pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
