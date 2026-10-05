#!/usr/bin/env python3
"""Cross-lingual SER audit: speaker-independent per-language probing.

One command to run the headline per-language probe (SUPERB-style weighted sum over
all layers, frozen encoder, speaker-independent 60/40 splits, multi-seed, 4-class)
for ANY self-supervised speech encoder, once its per-layer embeddings have been
extracted into a cache directory.

Usage:
    python audit/run.py --cache <embeddings_dir> --suffix <model_file_suffix>

  <model_file_suffix> is the filename stem used at extraction time. For example,
  cache files named '<dataset>__facebook__hubert-base-ls960.npy/.csv' have suffix
  'facebook__hubert-base-ls960'. Extract a new encoder with scripts/extract_embeddings.py.

Each dataset's cache is a per-layer array (N, L, D) plus a metadata CSV with columns
'speaker_id' and 'emotion'; emotions are restricted here to the 4-class intersection
{angry, happy, neutral, sad}.

This covers the per-language arm of the audit (the headline result). For the
cross-lingual transfer and the INLP speaker-confound control, see
analysis/transfer_superb.py and analysis/confound_analysis.py.

This mirrors analysis/probe_superb.py exactly (same split, labels, probe settings);
smoke-test it once against the provided cached embeddings before trusting it on a new
encoder, and adjust the metadata column names below if your extractor differs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# audit/ sits next to analysis/ in the repo; reuse the shared probe.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from superb_probe import SuperbProbe  # noqa: E402

EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
DATASETS = ["emodb", "ravdess", "crema_d", "esd_zh", "subesco"]
SEEDS = [20260505, 20260506, 20260507, 20260508, 20260509]


def load(cache: Path, dataset: str, suffix: str):
    base = cache / f"{dataset}__{suffix}"
    emb = np.load(str(base) + ".npy")
    meta = pd.read_csv(str(base) + ".csv")
    return emb, meta


def speaker_split(meta: pd.DataFrame, seed: int, train_frac: float = 0.6):
    rng = np.random.default_rng(seed)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train = set(speakers[:n_train])
    mask = meta["speaker_id"].isin(train).to_numpy()
    return mask, ~mask


def probe_dataset(emb: np.ndarray, meta: pd.DataFrame, seed: int, epochs: int) -> float:
    keep = meta["emotion"].isin(EMOTION_4CLASS).to_numpy()
    emb, meta = emb[keep], meta.loc[keep].reset_index(drop=True)
    if emb.ndim != 3:
        raise ValueError(f"expected per-layer (N, L, D) embeddings, got {emb.shape}")
    tr, te = speaker_split(meta, seed)
    lab = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    ytr = np.array([lab[y] for y in meta.loc[tr, "emotion"]])
    yte = np.array([lab[y] for y in meta.loc[te, "emotion"]])
    probe = SuperbProbe(
        n_layers=emb.shape[1], dim=emb.shape[2], n_classes=len(EMOTION_4CLASS),
        lr=1e-3, weight_decay=1e-4, epochs=epochs, batch_size=128, seed=seed,
    )
    probe.fit(emb[tr], ytr)
    pred = probe.predict(emb[te])
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser(description="Per-language SER probing audit.")
    ap.add_argument("--cache", required=True, type=Path,
                    help="directory holding <dataset>__<suffix>.npy / .csv")
    ap.add_argument("--suffix", required=True,
                    help="model filename suffix used at extraction")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()

    print(f"# Per-language probing audit for '{args.suffix}'")
    print(f"# {len(args.seeds)} seeds, speaker-independent 60/40 splits, 4-class\n")
    print(f"{'dataset':10s} {'mean':>7s} {'std':>6s}")
    for d in args.datasets:
        emb, meta = load(args.cache, d, args.suffix)
        accs = [probe_dataset(emb, meta, s, args.epochs) for s in args.seeds]
        print(f"{d:10s} {np.mean(accs):7.3f} {np.std(accs):6.3f}")


if __name__ == "__main__":
    main()
