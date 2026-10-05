"""Extract per-layer mean-pooled SSL embeddings for one (model, dataset) pair.

Usage:
    python extract_embeddings.py --model facebook/hubert-base-ls960 --dataset emodb

Outputs (to /path/to/data/embeddings/):
    <dataset>__<model_safe>.npy        # (N, L, D) float32 array of per-layer
                                       # mean-pooled embeddings. L = num_layers + 1
                                       # (input embedding + transformer layers).
                                       # Last-layer view: arr[:, -1, :].
    <dataset>__<model_safe>.csv        # metadata: row -> dataset, audio_path, emotion, speaker, original_emotion

Resamples to 16 kHz mono before feeding the model.
Saves all hidden states (input embedding + every transformer layer), so future
analyses can pick any layer without re-extraction.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import librosa
import torch
from transformers import AutoModel, AutoFeatureExtractor

WORKDIR = Path("/path/to/data")
CACHE_DIR = WORKDIR / "models" / "hf-cache"
EMB_DIR = WORKDIR / "embeddings"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
sys.path.insert(0, str(WORKDIR / "scripts"))
from datasets import load_dataset as load_corpus  # local module


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model id, e.g. facebook/hubert-base-ls960")
    ap.add_argument("--dataset", required=True, help="Corpus name, e.g. emodb")
    ap.add_argument("--batch-size", type=int, default=8, help="Audio samples per forward pass")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N records (0 = all)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device} model={args.model} dataset={args.dataset}")

    # Load corpus records
    records = load_corpus(args.dataset)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"[setup] {len(records)} records to process")

    # Load model
    t0 = time.time()
    fe = AutoFeatureExtractor.from_pretrained(args.model, cache_dir=str(CACHE_DIR))
    model = AutoModel.from_pretrained(args.model, cache_dir=str(CACHE_DIR)).to(device)
    model.eval()
    print(f"[setup] model loaded in {time.time() - t0:.2f}s")

    embeddings = []
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for batch_start in range(0, len(records), args.batch_size):
        batch_records = records[batch_start:batch_start + args.batch_size]
        audios = []
        for rec in batch_records:
            audio, _ = librosa.load(rec["audio_path"], sr=16000, mono=True)
            audios.append(audio)

        inputs = fe(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        attn = inputs.get("attention_mask")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states is a tuple of length L+1: input embedding + L transformer layers
        # Each tensor: (B, T, D)
        hidden_states = outputs.hidden_states  # tuple of (B, T, D)

        # Build attention mask aligned to the model's frame rate
        if attn is not None and attn.shape[1] == hidden_states[0].shape[1]:
            mask = attn.to(device).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled_per_layer = [
                (h * mask).sum(dim=1) / denom for h in hidden_states
            ]
        else:
            pooled_per_layer = [h.mean(dim=1) for h in hidden_states]

        # Stack into (B, L+1, D)
        pooled = torch.stack(pooled_per_layer, dim=1)
        embeddings.append(pooled.cpu().numpy().astype(np.float32))

        if (batch_start // args.batch_size) % 10 == 0:
            done = min(batch_start + args.batch_size, len(records))
            rate = done / (time.time() - t0)
            print(f"[progress] {done}/{len(records)}  ({rate:.1f} utt/s)")

    all_emb = np.concatenate(embeddings, axis=0)  # (N, L+1, D)
    print(f"[done] per-layer embeddings shape: {all_emb.shape}, total time: {time.time() - t0:.1f}s")

    # Save
    out_npy = EMB_DIR / f"{args.dataset}__{safe_name(args.model)}.npy"
    out_csv = EMB_DIR / f"{args.dataset}__{safe_name(args.model)}.csv"
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
    print(f"[saved] {out_npy}  shape={all_emb.shape}  dtype={all_emb.dtype}")
    print(f"[saved] {out_csv}  rows={len(records)}")


if __name__ == "__main__":
    main()
