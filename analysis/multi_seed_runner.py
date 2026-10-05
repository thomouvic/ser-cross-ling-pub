"""Multi-seed runs of the headline experiments.

Re-runs the three core experiments with multiple random seeds to estimate
training-time variance on top of test-set bootstrap CIs:
  1. RQ1 per-language probes (4-class, 5 datasets x 3 models)
  2. Claim 1 speaker-independent (7-class, 3 datasets x 3 models)
  3. Claim 2 Thai (1 dataset x 4 models incl. _plus_base)

Each (experiment, model, dataset) combo trained from scratch with each seed in
SEEDS. Per-seed test predictions captured; aggregate mean/std across seeds.

Resume support: skips combos already in the per-seed CSV.

Performance: embeddings loaded ONCE per (model, dataset) combo; all 5 seeds
reuse the cached arrays in memory. CLI flag --experiment selects rq1, claim1,
or claim2 so jobs can run in parallel.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import (load_embeddings, DATASETS, MODEL_ALIAS, MODEL_ALIAS_WITH_PLUS,
                   CACHE_DIR)
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [20260505, 20260506, 20260507, 20260508, 20260509]  # 5 seeds
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
FULL_DATASETS = ["emodb_full", "ravdess_full", "subesco_full"]


def speaker_split(meta: pd.DataFrame, seed: int, train_frac: float = 0.6
                   ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()
    return train_mask, ~train_mask


def load_full(dataset: str, model: str) -> tuple[np.ndarray, pd.DataFrame]:
    suffix = MODEL_ALIAS[model]
    base = CACHE_DIR / f"{dataset}__{suffix}"
    return np.load(str(base) + ".npy"), pd.read_csv(str(base) + ".csv")


def run_one(emb: np.ndarray, meta: pd.DataFrame, seed: int,
             classes: list, train_frac: float = 0.6,
             epochs: int = 200, batch_size: int = 128) -> dict:
    if emb.ndim == 2:
        emb = emb[:, np.newaxis, :]  # (N, 1, D) for utterance-only

    train_mask, test_mask = speaker_split(meta, seed=seed, train_frac=train_frac)
    Xtr, Xte = emb[train_mask], emb[test_mask]
    label_to_int = {e: i for i, e in enumerate(classes)}
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])
    yte = np.asarray([label_to_int[y] for y in meta.loc[test_mask, "emotion"]])

    probe = SuperbProbe(n_layers=emb.shape[1], dim=emb.shape[2],
                       n_classes=len(classes), lr=1e-3, weight_decay=1e-4,
                       epochs=epochs, batch_size=batch_size, seed=seed,
                       verbose=False)
    probe.fit(Xtr, ytr)
    pred = probe.predict(Xte)
    return {
        "n_test": int(test_mask.sum()),
        "n_test_speakers": int(meta.loc[test_mask, "speaker_id"].nunique()),
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def run_with_resume(experiment: str, jobs: list[tuple], out_csv: Path) -> None:
    """jobs: list of (model, dataset, classes, train_frac, epochs, batch_size).
    Runs each job for each seed, saves per-seed results, supports resume.
    Loads embeddings ONCE per (model, dataset) combo; all seeds reuse cached
    arrays."""
    if out_csv.exists():
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict(orient="records")
        done = {(r["experiment"], r["model"], r["dataset"], r["seed"])
                 for r in rows}
        print(f"[resume] {len(rows)} prior rows in {out_csv.name}", flush=True)
    else:
        rows, done = [], set()

    for (model, dataset, classes, train_frac, epochs, batch_size) in jobs:
        # Skip combo entirely if all seeds already done.
        remaining = [s for s in SEEDS
                      if (experiment, model, dataset, s) not in done]
        if not remaining:
            print(f"  [skip-combo] {experiment} | {dataset} x {model} (all seeds done)",
                  flush=True)
            continue
        # Load embeddings ONCE for this combo.
        try:
            print(f"  [load] {experiment} | {dataset} x {model}", flush=True)
            emb, meta = (load_full(dataset, model)
                         if dataset.endswith("_full")
                         else load_embeddings(dataset, model, layer=None))
            print(f"    emb shape={emb.shape}, n_meta={len(meta)}", flush=True)
        except FileNotFoundError as e:
            print(f"    MISSING: {e}", flush=True)
            continue
        for seed in remaining:
            print(f"  {experiment} | {dataset:14s} x {model:25s} | seed {seed}",
                  flush=True)
            try:
                res = run_one(emb, meta, seed, classes,
                              train_frac=train_frac,
                              epochs=epochs, batch_size=batch_size)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rows.append({
                "experiment": experiment, "model": model, "dataset": dataset,
                "seed": seed, **res,
            })
            print(f"    acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  "
                  f"n_test={res['n_test']}", flush=True)
            pd.DataFrame(rows).to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["rq1", "claim1", "claim2", "all"],
                         default="all",
                         help="Which experiment to run (default: all)")
    args = parser.parse_args()
    out_csv = RESULTS_DIR / "multi_seed_per_seed.csv"

    if args.experiment in ("rq1", "all"):
        rq1_jobs = [(m, d, EMOTION_4CLASS, 0.6, 200, 128)
                     for m in MODEL_ALIAS for d in DATASETS]
        print(f"\n=== RQ1 (per-language 4-class) — "
              f"{len(rq1_jobs)} combos × {len(SEEDS)} seeds ===", flush=True)
        run_with_resume("rq1", rq1_jobs, out_csv)

    if args.experiment in ("claim1", "all"):
        claim1_jobs = []
        for m in MODEL_ALIAS:
            for d in FULL_DATASETS:
                try:
                    _, meta = load_full(d, m)
                    cls = sorted(meta["emotion"].unique())
                    claim1_jobs.append((m, d, cls, 0.6, 200, 128))
                except FileNotFoundError:
                    print(f"[skip] no embedding for {d} x {m}", flush=True)
        print(f"\n=== Claim 1 (speaker-independent 7-class) — "
              f"{len(claim1_jobs)} combos × {len(SEEDS)} seeds ===", flush=True)
        run_with_resume("claim1_si", claim1_jobs, out_csv)

    if args.experiment in ("claim2", "all"):
        claim2_jobs = [(m, "thai_ser", EMOTION_4CLASS, 0.6, 100, 256)
                        for m in MODEL_ALIAS_WITH_PLUS]
        print(f"\n=== Claim 2 (Thai 4-class with _plus) — "
              f"{len(claim2_jobs)} combos × {len(SEEDS)} seeds ===", flush=True)
        run_with_resume("claim2_thai", claim2_jobs, out_csv)

    # Aggregate mean/std across seeds.
    if not out_csv.exists():
        print("No results to aggregate."); return
    df = pd.read_csv(out_csv)
    agg = (df.groupby(["experiment", "model", "dataset"])
             .agg(acc_mean=("accuracy", "mean"),
                  acc_std =("accuracy", "std"),
                  f1_mean =("macro_f1", "mean"),
                  f1_std  =("macro_f1", "std"),
                  n_seeds =("seed", "count"))
             .reset_index())
    agg_csv = RESULTS_DIR / "multi_seed_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"\nAggregated → {agg_csv}", flush=True)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
