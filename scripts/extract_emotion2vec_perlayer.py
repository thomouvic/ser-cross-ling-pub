"""Extract per-layer mean-pooled embeddings for emotion2vec_base.

Bypasses FunASR's granularity="utterance" pooling (which would give us a
single layer-averaged vector) and instead uses the underlying model's
extract_features() method, which returns layer_results: a list of per-layer
hidden states. We mean-pool each layer over time (frame axis) to produce
utterance-level per-layer embeddings: (N, n_layers + 1, D).

The +1 is the input embedding (the modality_encoders output) which we treat
as "layer 0" for SUPERB consistency. emotion2vec_base has 8 transformer
blocks, so the saved shape is (N, 9, 768).

Outputs:
    embeddings/<dataset>__emotion2vec-base.npy   shape (N, 9, 768)
    embeddings/<dataset>__emotion2vec-base.csv   metadata
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import librosa

WORKDIR = Path("/path/to/data/")
CACHE_DIR = WORKDIR / "models" / "hf-cache"
EMB_DIR = WORKDIR / "embeddings"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR)

import sys
sys.path.insert(0, str(WORKDIR / "scripts"))
from datasets import load_dataset as load_corpus

from funasr import AutoModel  # type: ignore

DEFAULT_MODEL_HF_ID = "emotion2vec/emotion2vec_base"
DEFAULT_MODEL_LABEL = "emotion2vec-base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_HF_ID,
                    help="HuggingFace model id (e.g., emotion2vec/emotion2vec_base or iic/emotion2vec_plus_base)")
    ap.add_argument("--model-label", default=DEFAULT_MODEL_LABEL,
                    help="Short label for output filename (e.g., emotion2vec-base, emotion2vec-plus-base)")
    args = ap.parse_args()
    MODEL_HF_ID = args.model_id
    MODEL_LABEL = args.model_label

    records = load_corpus(args.dataset)
    if args.limit > 0:
        records = records[:args.limit]
    print(f"[setup] {len(records)} records")

    t0 = time.time()
    funasr_model = AutoModel(model=MODEL_HF_ID, hub="hf", disable_update=True)
    inner = funasr_model.model  # the underlying torch Emotion2vec module
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inner.to(device)
    inner.eval()
    print(f"[setup] model loaded in {time.time() - t0:.2f}s; device={device}")

    EMB_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = []
    t0 = time.time()
    for i, rec in enumerate(records):
        # FunASR's preferred path: load + resample to 16k mono
        audio, _ = librosa.load(rec["audio_path"], sr=16000, mono=True)
        # The model expects shape (B, T) of raw audio at 16kHz
        x = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        with torch.no_grad():
            # mode=None lets the model figure it out; mask=False = inference time
            out = inner.extract_features(x, padding_mask=None, mask=False, remove_extra_tokens=True)
        # out["x"] is (B, T, D) for the final layer
        # out["layer_results"] is a list of tuples; per-layer hidden states
        # The exact format depends on the model; we'll grab the per-layer states.

        # Inspect on first iteration only
        if i == 0:
            n_layers = len(out["layer_results"])
            sample = out["layer_results"][0]
            print(f"[debug] layer_results length = {n_layers}, "
                  f"first entry type = {type(sample).__name__}, ", end="")
            if isinstance(sample, tuple):
                print(f"tuple of {len(sample)} elements; "
                      f"first elem shape = {sample[0].shape if torch.is_tensor(sample[0]) else 'not tensor'}")
            elif torch.is_tensor(sample):
                print(f"tensor shape = {sample.shape}")
            else:
                print(f"unknown: {sample}")

        # Each layer_result is typically a tuple (hidden_state, attn_weights, ...) where
        # hidden_state has shape (T, B, D) (transformer convention) or (B, T, D).
        # We collect per-layer states and the final out["x"] as the "topmost" representation.
        per_layer = []
        # Layer 0 conventionally is the input embedding; data2vec puts it as the FIRST
        # layer_result in some configs and as a separate field in others. We'll include
        # final output ("x") at the top, plus all intermediate states.
        for layer_state in out["layer_results"]:
            if isinstance(layer_state, tuple):
                hidden = layer_state[0]
            else:
                hidden = layer_state
            # Normalize to (B, T, D)
            if hidden.dim() == 3 and hidden.shape[0] != 1 and hidden.shape[1] == 1:
                # (T, B, D) -> (B, T, D)
                hidden = hidden.transpose(0, 1)
            elif hidden.dim() == 3 and hidden.shape[1] == 1 and hidden.shape[0] != 1:
                # ambiguous; assume (T, B, D)
                hidden = hidden.transpose(0, 1)
            # mean-pool over time (axis=1)
            pooled = hidden.mean(dim=1)  # (B, D)
            per_layer.append(pooled.squeeze(0).cpu().numpy())

        # Also include the final x (which may differ from the last layer_result)
        final_pooled = out["x"].mean(dim=1).squeeze(0).cpu().numpy()
        per_layer.append(final_pooled)

        embeddings.append(np.stack(per_layer, axis=0))  # (n_layers + 1, D)

        if (i + 1) % 200 == 0 or i + 1 == len(records):
            rate = (i + 1) / (time.time() - t0)
            print(f"[progress] {i + 1}/{len(records)}  ({rate:.1f} utt/s)")

    all_emb = np.stack(embeddings, axis=0).astype(np.float32)  # (N, n_layers + 1, D)
    print(f"[done] per-layer embeddings shape: {all_emb.shape}, total time: {time.time() - t0:.1f}s")

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
