"""Load cached embeddings + metadata from the local cache.

Cache layout (after rsync from a GPU cluster):
    embeddings_cache/<dataset>__<model_safe>.npy
    embeddings_cache/<dataset>__<model_safe>.csv

Where model_safe is the model id with '/' -> '__'.

Returns embeddings as numpy arrays:
  - HuBERT, WavLM:  (N, L+1, D)  per-layer mean-pooled (L=12, D=768, +1 input embed)
  - emotion2vec:    (N, D)       utterance-level (D=768)

Metadata is a pandas DataFrame indexed by row.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Default to a local embeddings_cache/ next to the repo; set DATA_CACHE to point
# the same analysis code at embeddings stored elsewhere (e.g. on a cluster)
# without editing paths.
CACHE_DIR = Path(os.environ.get(
    "DATA_CACHE",
    Path(__file__).resolve().parent.parent / "embeddings_cache",
))

DATASETS = ["emodb", "ravdess", "crema_d", "esd_zh", "subesco"]
DATASETS_EXTENDED = ["emodb", "ravdess", "crema_d", "esd_zh", "subesco", "thai_ser"]

MODEL_ALIAS = {
    "hubert":      "facebook__hubert-base-ls960",
    "wavlm":       "microsoft__wavlm-base",
    "emotion2vec": "emotion2vec-base",
}

# Extended set including the contaminated _plus variant for Claim 2 testing on Thai.
# Use these only when evaluating on a language NOT in EmoBox (Thai is the prime example).
MODEL_ALIAS_WITH_PLUS = {
    "hubert":      "facebook__hubert-base-ls960",
    "wavlm":       "microsoft__wavlm-base",
    "emotion2vec_base":      "emotion2vec-base",
    "emotion2vec_plus_base": "emotion2vec-plus-base",
}

DATASET_LANG = {
    "emodb":   "de",
    "ravdess": "en",
    "crema_d": "en",
    "esd_zh":  "zh",
    "subesco": "bn",
    "thai_ser": "th",
}


def load_embeddings(dataset: str, model: str, layer: int | str | None = None
                    ) -> tuple[np.ndarray, pd.DataFrame]:
    """Load (embeddings, metadata) for one (dataset, model) pair.

    layer:
      - None: return embeddings as-is. For HuBERT/WavLM that's (N, L+1, D);
              for emotion2vec it's already (N, D).
      - int: pick a specific layer index from per-layer arrays. -1 = last.
      - "mean_layers": average across the layer axis (only valid for per-layer).
    """
    # Accept extended (with-plus) aliases for Thai or any non-EmoBox dataset
    aliases = MODEL_ALIAS_WITH_PLUS if model not in MODEL_ALIAS and model in MODEL_ALIAS_WITH_PLUS else MODEL_ALIAS
    if model not in aliases:
        raise ValueError(f"unknown model {model!r}, expected one of {list(MODEL_ALIAS)} or {list(MODEL_ALIAS_WITH_PLUS)}")
    if dataset not in DATASETS and dataset not in DATASETS_EXTENDED and not dataset.endswith("_full"):
        raise ValueError(f"unknown dataset {dataset!r}")

    model_suffix = aliases[model]
    base = CACHE_DIR / f"{dataset}__{model_suffix}"
    emb = np.load(str(base) + ".npy")
    meta = pd.read_csv(str(base) + ".csv")

    if emb.ndim == 3 and layer is not None:
        if layer == "mean_layers":
            emb = emb.mean(axis=1)
        else:
            emb = emb[:, layer, :]

    # add language column for convenience
    meta["language"] = DATASET_LANG[dataset]

    return emb, meta


def load_all(model: str, layer: int | str | None = -1
             ) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate embeddings + metadata across all 5 datasets for one model.

    For HuBERT/WavLM layer defaults to -1 (last transformer layer).
    For emotion2vec layer is ignored.
    """
    embs, metas = [], []
    for d in DATASETS:
        e, m = load_embeddings(d, model, layer=layer)
        embs.append(e)
        metas.append(m)
    full_emb = np.concatenate(embs, axis=0)
    full_meta = pd.concat(metas, ignore_index=True)
    return full_emb, full_meta


if __name__ == "__main__":
    print("=== inventory ===")
    for d in DATASETS:
        for m in MODEL_ALIAS:
            try:
                e, meta = load_embeddings(d, m)
                print(f"  {d:8s} x {m:11s}  emb={e.shape}  meta={len(meta)}")
            except FileNotFoundError as err:
                print(f"  {d:8s} x {m:11s}  MISSING")
