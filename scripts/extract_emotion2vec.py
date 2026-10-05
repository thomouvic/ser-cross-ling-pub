"""Extract utterance-level emotion2vec embeddings for one dataset.

Usage:
    python extract_emotion2vec.py --dataset emodb

Outputs:
    <dataset>__emotion2vec-plus-base.npy   # (N, D) float32 — utterance-level
    <dataset>__emotion2vec-plus-base.csv   # metadata

Note: emotion2vec only exports utterance-level embedding via the FunASR API,
not per-layer. So this file's shape is (N, D), unlike the HuBERT/WavLM files
which are (N, L, D). Layer-wise analyses are restricted to HuBERT/WavLM.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

WORKDIR = Path("/path/to/data/")
CACHE_DIR = WORKDIR / "models" / "hf-cache"
EMB_DIR = WORKDIR / "embeddings"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)
os.environ["MODELSCOPE_CACHE"] = str(WORKDIR / "models" / "modelscope-cache")

import sys
sys.path.insert(0, str(WORKDIR / "scripts"))
from datasets import load_dataset as load_corpus

from funasr import AutoModel  # type: ignore

MODEL_HF_ID = "emotion2vec/emotion2vec_base"
MODEL_LABEL = "emotion2vec-base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_corpus(args.dataset)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"[setup] {len(records)} records")

    t0 = time.time()
    model = AutoModel(model=MODEL_HF_ID, hub="hf", disable_update=True)
    print(f"[setup] model loaded in {time.time() - t0:.2f}s")

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = []
    t0 = time.time()
    for i, rec in enumerate(records):
        # FunASR's generate handles loading + resampling internally
        res = model.generate(
            input=rec["audio_path"],
            granularity="utterance",
            extract_embedding=True,
            disable_pbar=True,
        )
        # res is a list of dicts; "feats" is the utterance embedding (numpy array)
        feats = np.asarray(res[0]["feats"], dtype=np.float32)
        embeddings.append(feats)
        if (i + 1) % 200 == 0 or i + 1 == len(records):
            rate = (i + 1) / (time.time() - t0)
            print(f"[progress] {i + 1}/{len(records)} ({rate:.1f} utt/s)")

    all_emb = np.stack(embeddings, axis=0)
    print(f"[done] embeddings shape: {all_emb.shape}, total time: {time.time() - t0:.1f}s")

    out_npy = EMB_DIR / f"{args.dataset}__{MODEL_LABEL}.npy"
    out_csv = EMB_DIR / f"{args.dataset}__{MODEL_LABEL}.csv"
    np.save(out_npy, all_emb)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["row", "dataset", "audio_path", "emotion", "speaker_id", "original_emotion"])
        writer.writeheader()
        for i, rec in enumerate(records):
            writer.writerow({
                "row": i,
                "dataset": rec["dataset"],
                "audio_path": rec["audio_path"],
                "emotion": rec["emotion"],
                "speaker_id": rec["speaker_id"],
                "original_emotion": rec["original_emotion"],
            })
    print(f"[saved] {out_npy} {all_emb.shape} {all_emb.dtype}")
    print(f"[saved] {out_csv} rows={len(records)}")


if __name__ == "__main__":
    main()
