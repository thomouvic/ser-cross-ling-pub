"""Speaker-DISJOINT cross-lingual K-shot transfer.

Rationale: in the original few-shot protocol
(``fewshot_transfer.py`` / ``fewshot_multi_trial.py``) the K target-language
shots are drawn from ALL target speakers and the test set is simply the
complement, so a shot from speaker X and test utterances from speaker X can
coexist. That is utterance-disjoint but NOT speaker-disjoint, and can
overestimate few-shot adaptation (the probe sees a few labeled utterances from
speakers it is then tested on).

This script reruns the sweep with speaker-disjoint shot sampling:
  - Target speakers are split into a POOL set (from which shots are drawn) and
    a TEST set. The two speaker sets are disjoint.
  - The evaluation set is fixed to the TEST speakers' utterances across all K, so
    K=0 and K>0 are measured on the identical speaker-independent test set.
  - K>0 shots are sampled (emotion-stratified) only from POOL speakers, so no
    shot speaker ever appears in the test set.

Source-language training is unchanged: the source-language train split from
``in_language_split`` (train_frac=0.6, seeded), exactly as the original protocol.

The pool/test speaker split is fixed (seeded once), and only the shot sampling
varies across the 3 trials, mirroring the original protocol's variance structure
(fixed source split, varied shots). This keeps the before/after comparison clean.

Outputs:
  results/fewshot_disjoint_per_trial.csv
  results/fewshot_disjoint_summary.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewshot_transfer import (
    language_data, in_language_split, stratified_sample_by_emotion,
    EMOTION_4CLASS, LANGS, K_VALUES, RNG_SEED, PROBE_EPOCHS,
)
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_TRIALS = 3
MODELS = ["hubert", "wavlm", "emotion2vec"]
POOL_FRAC = 0.4          # fraction of target speakers reserved as the shot pool
SPLIT_SEED = RNG_SEED    # seed for the (fixed) pool/test speaker split


def target_speaker_split(meta_tgt: pd.DataFrame, pool_frac: float, seed: int):
    """Split target speakers into a disjoint POOL (shots) and TEST set.

    Returns (pool_mask, test_mask) over utterances, plus the speaker lists.
    """
    rng = np.random.default_rng(seed)
    speakers = sorted(meta_tgt["speaker_id"].unique())
    rng.shuffle(speakers)
    n_pool = max(1, int(round(pool_frac * len(speakers))))
    # Guarantee at least one test speaker.
    n_pool = min(n_pool, len(speakers) - 1)
    pool_speakers = set(speakers[:n_pool])
    test_speakers = set(speakers[n_pool:])
    pool_mask = meta_tgt["speaker_id"].isin(pool_speakers).to_numpy()
    test_mask = meta_tgt["speaker_id"].isin(test_speakers).to_numpy()
    return pool_mask, test_mask, sorted(pool_speakers), sorted(test_speakers)


def transfer_disjoint(model: str, src: str, tgt: str, data: dict, K: int,
                      shot_seed: int) -> float:
    """Train probe on source-lang train split + K speaker-disjoint target shots.
    Evaluate on the fixed disjoint target TEST speakers. Returns accuracy."""
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}

    # Source training data (unchanged from the original protocol).
    Xsrc_full, meta_src = data[src]
    src_train_mask, _ = in_language_split(meta_src)
    Xsrc = Xsrc_full[src_train_mask]
    ysrc = np.asarray([label_to_int[y] for y in meta_src.loc[src_train_mask, "emotion"]])

    # Target: disjoint pool/test speaker split (fixed across trials).
    Xtgt_full, meta_tgt = data[tgt]
    pool_mask, test_mask, _, _ = target_speaker_split(meta_tgt, POOL_FRAC, SPLIT_SEED)
    test_idx = np.where(test_mask)[0]

    if K > 0:
        rng = np.random.default_rng(shot_seed)
        shot_idx = stratified_sample_by_emotion(pool_mask, meta_tgt, K, rng)
        # Safety: shots must lie strictly inside the pool (never in test).
        assert not test_mask[shot_idx].any(), "shot leaked into test speakers"
        Xshots = Xtgt_full[shot_idx]
        yshots = np.asarray([label_to_int[meta_tgt.iloc[i]["emotion"]] for i in shot_idx])
        X_train = np.concatenate([Xsrc, Xshots], axis=0)
        y_train = np.concatenate([ysrc, yshots], axis=0)
        n_shots = int(len(shot_idx))
    else:
        X_train, y_train = Xsrc, ysrc
        n_shots = 0

    Xtest = Xtgt_full[test_idx]
    ytest = np.asarray([label_to_int[meta_tgt.iloc[i]["emotion"]] for i in test_idx])

    n_layers = X_train.shape[1]
    dim = X_train.shape[2]
    probe = SuperbProbe(n_layers=n_layers, dim=dim, n_classes=len(EMOTION_4CLASS),
                        lr=1e-3, weight_decay=1e-4, epochs=PROBE_EPOCHS,
                        batch_size=256, verbose=False, seed=shot_seed)
    probe.fit(X_train, y_train)
    pred = probe.predict(Xtest)
    return accuracy_score(ytest, pred), n_shots, int(len(test_idx))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    args = parser.parse_args()
    selected_models = MODELS if args.model == "all" else [args.model]

    out_csv = RESULTS_DIR / "fewshot_disjoint_per_trial.csv"
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
                        shot_seed = RNG_SEED + trial * 1000 + K
                        acc, n_shots, n_test = transfer_disjoint(
                            model, src, tgt, data, K, shot_seed)
                        rows.append({"model": model, "src": src, "tgt": tgt,
                                     "K": K, "trial": trial, "seed": shot_seed,
                                     "accuracy": float(acc), "n_shots": n_shots,
                                     "n_test": n_test})
                        print(f"  {src}->{tgt}  K={K:3d}  trial={trial}  "
                              f"acc={acc:.4f}  (shots={n_shots}, test={n_test})",
                              flush=True)
                        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.read_csv(out_csv)
    agg = (df.groupby(["model", "src", "tgt", "K"])
             .agg(mean_acc=("accuracy", "mean"),
                  std_acc=("accuracy", "std"),
                  n_trials=("trial", "count"))
             .reset_index())
    agg["std_acc"] = agg["std_acc"].fillna(0.0)
    agg_csv = RESULTS_DIR / "fewshot_disjoint_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"\nAggregated -> {agg_csv}", flush=True)

    # Print the headline before/after-ready table: mean across the 12 pairs per K.
    print("\n=== mean accuracy across 12 pairs (speaker-disjoint) ===", flush=True)
    per_model_K = (agg.groupby(["model", "K"])["mean_acc"].mean().reset_index())
    for model in selected_models:
        sub = per_model_K[per_model_K.model == model].sort_values("K")
        cells = "  ".join(f"K={int(r.K)}:{r.mean_acc:.3f}" for r in sub.itertuples())
        print(f"  {model:12s}  {cells}", flush=True)


if __name__ == "__main__":
    main()
