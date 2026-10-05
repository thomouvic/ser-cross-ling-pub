"""Speaker-disjoint cross-lingual K-shot transfer to and from Thai.

Adds Thai as a fifth transfer language alongside the main 12-pair matrix over
the four EmoBox languages (fewshot_speaker_disjoint.py), which is left unchanged:
  - Thai as target: en->th, de->th, zh->th, bn->th
  - Thai as source: th->en, th->de, th->zh, th->bn

Protocol is identical to ``fewshot_speaker_disjoint.py`` (same function, same
seeds, same pool/test speaker split rule, same probe): source training on the
60% train-speaker split, K emotion-stratified shots drawn only from a disjoint
target speaker pool, evaluation on the fixed disjoint target test speakers.

Thai uses the same cached 4-class subset as the in-language Thai result
(THAI-SER, agreement >= 0.5, "frustrated" excluded, 13,132 utterances).

Before the sweep, one known main-matrix cell (en->de, K=0) is recomputed
and compared with results/fewshot_disjoint_summary.csv as a reproducibility
check of the environment.

Usage (one model per job, so jobs can run in parallel without CSV clashes):
  python analysis/fewshot_thai_transfer.py --model hubert
  python analysis/fewshot_thai_transfer.py --aggregate

Outputs:
  results/fewshot_thai_per_trial_<model>.csv
  results/fewshot_thai_summary.csv   (written by --aggregate)
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewshot_transfer import language_data, LANGS, K_VALUES, RNG_SEED
from fewshot_speaker_disjoint import transfer_disjoint, N_TRIALS, MODELS
from load import load_embeddings

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
THAI = "th"
PAIRS = [(l, THAI) for l in LANGS] + [(THAI, l) for l in LANGS]


def data_with_thai(model: str) -> dict:
    data = language_data(model)
    emb, meta = load_embeddings("thai_ser", model, layer=None)
    data[THAI] = (emb, meta.reset_index(drop=True))
    return data


def repro_check(model: str, data: dict) -> None:
    """Recompute one main-matrix cell and compare with the stored value."""
    ref = pd.read_csv(RESULTS_DIR / "fewshot_disjoint_summary.csv")
    row = ref[(ref.model == model) & (ref.src == "en") & (ref.tgt == "de") & (ref.K == 0)]
    if row.empty:
        print("[repro] no reference row found", flush=True)
        return
    expected = float(row.mean_acc.iloc[0])
    acc, _, _ = transfer_disjoint(model, "en", "de", data, 0, RNG_SEED)
    status = "MATCH" if abs(acc - expected) < 1e-6 else "MISMATCH"
    print(f"[repro] {model} en->de K=0: got {acc:.6f}, stored value {expected:.6f} "
          f"-> {status}", flush=True)


def run_model(model: str) -> None:
    out_csv = RESULTS_DIR / f"fewshot_thai_per_trial_{model}.csv"
    if out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict(orient="records")
        done = {(r["src"], r["tgt"], int(r["K"]), int(r["trial"])) for r in rows}
        print(f"[resume] {len(rows)} prior rows in {out_csv.name}", flush=True)
    else:
        rows, done = [], set()

    print(f"\n=== {model} ===", flush=True)
    data = data_with_thai(model)
    repro_check(model, data)

    for src, tgt in PAIRS:
        for K in K_VALUES:
            n_t = 1 if K == 0 else N_TRIALS
            for trial in range(n_t):
                if (src, tgt, K, trial) in done:
                    continue
                shot_seed = RNG_SEED + trial * 1000 + K
                acc, n_shots, n_test = transfer_disjoint(model, src, tgt, data, K, shot_seed)
                rows.append({"model": model, "src": src, "tgt": tgt, "K": K,
                             "trial": trial, "seed": shot_seed, "accuracy": float(acc),
                             "n_shots": n_shots, "n_test": n_test})
                print(f"  {src}->{tgt}  K={K:3d}  trial={trial}  acc={acc:.4f}  "
                      f"(shots={n_shots}, test={n_test})", flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)


def aggregate() -> None:
    frames = [pd.read_csv(p) for p in sorted(RESULTS_DIR.glob("fewshot_thai_per_trial_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    agg = (df.groupby(["model", "src", "tgt", "K"])
             .agg(mean_acc=("accuracy", "mean"), std_acc=("accuracy", "std"),
                  n_trials=("trial", "count"))
             .reset_index())
    agg["std_acc"] = agg["std_acc"].fillna(0.0)
    out = RESULTS_DIR / "fewshot_thai_summary.csv"
    agg.to_csv(out, index=False)
    print(f"Aggregated -> {out}", flush=True)
    for direction, mask in [("Thai as target", agg.tgt == THAI), ("Thai as source", agg.src == THAI)]:
        print(f"\n=== mean accuracy, {direction} (4 pairs) ===", flush=True)
        sub = agg[mask].groupby(["model", "K"])["mean_acc"].mean().reset_index()
        for model in MODELS:
            s = sub[sub.model == model].sort_values("K")
            cells = "  ".join(f"K={int(r.K)}:{r.mean_acc:.3f}" for r in s.itertuples())
            print(f"  {model:12s}  {cells}", flush=True)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--aggregate", action="store_true")
    a = p.parse_args()
    if a.aggregate:
        aggregate()
    elif a.model:
        run_model(a.model)
    else:
        p.error("pass --model or --aggregate")


if __name__ == "__main__":
    main()
