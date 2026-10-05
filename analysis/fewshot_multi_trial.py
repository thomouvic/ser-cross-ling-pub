"""Multi-trial cross-lingual K-shot transfer.

Runs N_TRIALS=3 random shot samples per (model, src, tgt, K) cell, with K=0
remaining single-trial since it does not depend on shot sampling. Stores
per-trial rows for proper resume; aggregates at the end.

Reads in fewshot_transfer.py's existing logic for transfer_with_k_shots.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import DATASETS, MODEL_ALIAS, DATASET_LANG
from fewshot_transfer import (
    language_data, transfer_with_k_shots, EMOTION_4CLASS, LANGS,
    K_VALUES, RNG_SEED,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_TRIALS = 3
MODELS = ["hubert", "wavlm", "emotion2vec"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    args = parser.parse_args()

    selected_models = MODELS if args.model == "all" else [args.model]
    out_csv = RESULTS_DIR / "fewshot_multi_trial_per_trial.csv"

    if out_csv.exists():
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict(orient="records")
        done = {(r["model"], r["src"], r["tgt"], int(r["K"]), int(r["trial"]))
                for r in rows}
        print(f"[resume] {len(rows)} prior rows in {out_csv.name}", flush=True)
    else:
        rows, done = [], set()

    for model in selected_models:
        print(f"\n=== {model} ===", flush=True)
        # Skip whole model if all combos done.
        all_keys = []
        for src in LANGS:
            for tgt in LANGS:
                if src == tgt:
                    continue
                for K in K_VALUES:
                    n_t = 1 if K == 0 else N_TRIALS
                    for t in range(n_t):
                        all_keys.append((model, src, tgt, K, t))
        if all(k in done for k in all_keys):
            print(f"  [skip-model] all combos done", flush=True)
            continue
        data = language_data(model)
        for src in LANGS:
            for tgt in LANGS:
                if src == tgt:
                    continue
                for K in K_VALUES:
                    n_t = 1 if K == 0 else N_TRIALS
                    for trial in range(n_t):
                        if (model, src, tgt, K, trial) in done:
                            continue
                        # Distinct seed per trial; matches original script's K=0
                        # seed for trial 0 to keep that result identical.
                        seed = RNG_SEED + trial * 1000 + K
                        acc = transfer_with_k_shots(model, src, tgt, data, K, seed)
                        rows.append({"model": model, "src": src, "tgt": tgt,
                                      "K": K, "trial": trial, "seed": seed,
                                      "accuracy": float(acc)})
                        print(f"  {src}->{tgt}  K={K:3d}  trial={trial}  acc={acc:.4f}",
                              flush=True)
                        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Aggregate
    if not out_csv.exists():
        print("Nothing to aggregate."); return
    df = pd.read_csv(out_csv)
    agg = (df.groupby(["model", "src", "tgt", "K"])
             .agg(mean_acc=("accuracy", "mean"),
                  std_acc=("accuracy", "std"),
                  n_trials=("trial", "count"))
             .reset_index())
    agg["std_acc"] = agg["std_acc"].fillna(0.0)
    agg_csv = RESULTS_DIR / "fewshot_multi_trial_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"\nAggregated → {agg_csv}", flush=True)


if __name__ == "__main__":
    main()
