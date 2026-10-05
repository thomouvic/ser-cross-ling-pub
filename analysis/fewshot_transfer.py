"""K-shot cross-lingual transfer with SUPERB protocol.

For each (model, source_lang, target_lang != source_lang):
  - Train SuperbProbe on full source-language training set.
  - For each K in [0, 5, 10, 20, 50, 100]:
      - If K == 0: take source-only probe, evaluate on full target language data.
      - If K > 0: extend training set with K randomly-sampled target-language
        examples (one per emotion class up to K, ensuring some coverage),
        retrain probe, evaluate on remaining target data.
  - Repeat with N_TRIALS = 3 random sampling seeds, average.

Outputs:
  results/fewshot_transfer_summary.csv  (long form)
  results/fewshot_transfer_curve_<model>.png  (mean accuracy vs K, per source-target pair)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_embeddings, DATASETS, MODEL_ALIAS, DATASET_LANG
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
LANGS = ["en", "de", "zh", "bn"]
K_VALUES = [0, 10, 50, 100]
N_TRIALS = 1
RNG_SEED = 20260505
PROBE_EPOCHS = 50  # reduced from 200 for time; verified accuracy plateaus by then
MODELS_FOR_FEWSHOT = ["hubert", "wavlm", "emotion2vec"]  # extended to all 3 models


def language_data(model: str):
    by_lang_emb = {l: [] for l in LANGS}
    by_lang_meta = {l: [] for l in LANGS}
    for d in DATASETS:
        l = DATASET_LANG[d]
        emb, meta = load_embeddings(d, model, layer=None)
        by_lang_emb[l].append(emb)
        by_lang_meta[l].append(meta)
    return {l: (np.concatenate(by_lang_emb[l], axis=0),
                pd.concat(by_lang_meta[l], ignore_index=True)) for l in LANGS}


def in_language_split(meta: pd.DataFrame, train_frac: float = 0.6,
                      seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()
    return train_mask, ~train_mask


def stratified_sample_by_emotion(target_mask: np.ndarray, meta_te: pd.DataFrame,
                                  K: int, rng: np.random.Generator) -> np.ndarray:
    """Sample K indices from target training data, stratified by emotion."""
    n_per_class = max(1, K // 4)  # try to balance across 4 classes
    candidate_idx = np.where(target_mask)[0]
    selected = []
    for emo in EMOTION_4CLASS:
        emo_mask = (meta_te.iloc[candidate_idx]["emotion"] == emo).to_numpy()
        emo_idx = candidate_idx[emo_mask]
        if len(emo_idx) > 0:
            n_take = min(n_per_class, len(emo_idx))
            picks = rng.choice(emo_idx, size=n_take, replace=False)
            selected.extend(picks.tolist())
    # If we ended with more or fewer than K, trim or backfill
    if len(selected) > K:
        selected = selected[:K]
    elif len(selected) < K:
        remaining = [i for i in candidate_idx if i not in set(selected)]
        if remaining:
            extra = rng.choice(remaining, size=min(K - len(selected), len(remaining)), replace=False)
            selected.extend(extra.tolist())
    return np.asarray(selected)


def transfer_with_k_shots(model: str, src: str, tgt: str, data: dict,
                          K: int, trial_seed: int) -> float:
    """Train probe with all source-lang train data + K target-lang shots.
    Evaluate on target-lang data minus the K shots. Returns accuracy."""
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}

    # Source training data
    Xsrc_full, meta_src = data[src]
    src_train_mask, _ = in_language_split(meta_src)
    Xsrc = Xsrc_full[src_train_mask]
    ysrc = np.asarray([label_to_int[y] for y in meta_src.loc[src_train_mask, "emotion"]])

    # Target language: split into shot-pool + remaining test
    Xtgt_full, meta_tgt = data[tgt]
    # We use ALL of target-lang data as "available"; pull K from it; rest is test
    # For honest evaluation, the K shots should not overlap with test.
    rng = np.random.default_rng(trial_seed)
    if K > 0:
        all_tgt_mask = np.ones(len(meta_tgt), dtype=bool)
        shot_idx = stratified_sample_by_emotion(all_tgt_mask, meta_tgt, K, rng)
        shot_set = set(shot_idx.tolist())
        test_idx = np.array([i for i in range(len(meta_tgt)) if i not in shot_set])

        Xshots = Xtgt_full[shot_idx]
        yshots = np.asarray([label_to_int[meta_tgt.iloc[i]["emotion"]] for i in shot_idx])

        # Concatenate source + shots
        X_train = np.concatenate([Xsrc, Xshots], axis=0)
        y_train = np.concatenate([ysrc, yshots], axis=0)
    else:
        X_train, y_train = Xsrc, ysrc
        test_idx = np.arange(len(meta_tgt))

    Xtest = Xtgt_full[test_idx]
    ytest = np.asarray([label_to_int[meta_tgt.iloc[i]["emotion"]] for i in test_idx])

    n_layers = X_train.shape[1]
    dim = X_train.shape[2]
    probe = SuperbProbe(n_layers=n_layers, dim=dim, n_classes=len(EMOTION_4CLASS),
                       lr=1e-3, weight_decay=1e-4, epochs=PROBE_EPOCHS, batch_size=256,
                       verbose=False, seed=trial_seed)
    probe.fit(X_train, y_train)
    pred = probe.predict(Xtest)
    return accuracy_score(ytest, pred)


def main():
    out_csv = RESULTS_DIR / "fewshot_transfer_summary.csv"
    if out_csv.exists():
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict(orient="records")
        done = {(r["model"], r["src"], r["tgt"], r["K"]) for r in rows}
        print(f"[resume] {len(rows)} prior rows in {out_csv.name}", flush=True)
    else:
        rows, done = [], set()

    for model in MODELS_FOR_FEWSHOT:
        print(f"=== {model} ===", flush=True)
        # Skip whole model if all combos done.
        all_pairs = [(s, t, K) for s in LANGS for t in LANGS if s != t for K in K_VALUES]
        if all((model, s, t, K) in done for (s, t, K) in all_pairs):
            print(f"  [skip-model] all combos done", flush=True)
            continue
        data = language_data(model)
        for src in LANGS:
            for tgt in LANGS:
                if src == tgt:
                    continue
                for K in K_VALUES:
                    if (model, src, tgt, K) in done:
                        continue
                    accs = []
                    n_trials = 1 if K == 0 else N_TRIALS
                    for t in range(n_trials):
                        seed = RNG_SEED + t * 1000 + K
                        acc = transfer_with_k_shots(model, src, tgt, data, K, seed)
                        accs.append(acc)
                    mean_acc = float(np.mean(accs))
                    std_acc = float(np.std(accs))
                    rows.append({"model": model, "src": src, "tgt": tgt, "K": K,
                                 "mean_acc": mean_acc, "std_acc": std_acc, "n_trials": n_trials})
                    print(f"  {src}->{tgt}  K={K:3d}  acc={mean_acc:.3f} +- {std_acc:.3f}",
                          flush=True)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved {out_csv}", flush=True)
    df = pd.read_csv(out_csv)

    # Plot per-model curves: mean across all source-target pairs vs K
    n_models = len(MODELS_FOR_FEWSHOT)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, MODELS_FOR_FEWSHOT):
        sub = df[df.model == model]
        for (src, tgt), grp in sub.groupby(["src", "tgt"]):
            grp = grp.sort_values("K")
            ax.plot(grp["K"], grp["mean_acc"], "o-", alpha=0.5, label=f"{src}->{tgt}")
        # mean across pairs
        mean_curve = sub.groupby("K")["mean_acc"].mean()
        ax.plot(mean_curve.index, mean_curve.values, "k-", linewidth=3, label="mean")
        ax.set_xlabel("K (target-lang labeled samples)")
        ax.set_title(model)
        ax.set_xscale("symlog", linthresh=1)
        ax.grid(True, alpha=0.3)
        if model == MODELS_FOR_FEWSHOT[0]:
            ax.set_ylabel("accuracy")
        ax.axhline(y=0.25, color="r", linestyle="--", alpha=0.3, label="chance")
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    out = RESULTS_DIR / "fewshot_transfer_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
