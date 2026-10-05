"""Held-out speaker decodability after INLP.

Re-runs the INLP pipeline as in confound_analysis.py, but additionally measures
linear speaker decodability on the 62 *held-out* speakers (who were not seen
when fitting the speaker classifier during INLP). The question this addresses:
does INLP generalize to remove speaker information for speakers it never trained
the speaker-classifier on, or is the removal only certified on the train-portion?

For the held-out portion, we report two numbers:
  (a) "Pre" — linear speaker accuracy on a within-test-portion stratified
      classifier (62-class) on the pre-INLP weighted-sum representation.
      This is the upper-bound: how much speaker information is in the
      representation before any cleaning.
  (b) "Post" — same classifier on the post-INLP test-portion. The drop from
      (a) to (b) measures how much INLP removed for held-out speakers.

Within-test-portion classifier: 70/30 utterance-level split among test speakers'
utterances; 62-class softmax classifier; report test-set accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import load_embeddings, DATASETS, MODEL_ALIAS, DATASET_LANG
from superb_probe import SuperbProbe

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
RNG_SEED = 20260505


def speaker_split(meta, train_frac=0.6, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()
    return train_mask, ~train_mask


def get_superb_representation(model: str):
    embs, metas = [], []
    for d in DATASETS:
        e, m = load_embeddings(d, model, layer=None)
        m["language"] = DATASET_LANG[d]
        embs.append(e); metas.append(m)
    X = np.concatenate(embs, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    train_mask, _ = speaker_split(meta, train_frac=0.8)
    Xtr = X[train_mask]
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])
    probe = SuperbProbe(n_layers=X.shape[1], dim=X.shape[2],
                        n_classes=len(EMOTION_4CLASS),
                        lr=1e-3, weight_decay=1e-4, epochs=100, batch_size=256,
                        seed=RNG_SEED)
    probe.fit(Xtr, ytr)
    Xws = probe.weighted_sum(X)
    return Xws, meta


def project_null_space(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank = (S > 1e-6).sum()
    Vrow = Vt[:rank]
    return X - X @ Vrow.T @ Vrow


def within_speaker_split(meta_subset: pd.DataFrame, train_frac: float = 0.7,
                          seed: int = RNG_SEED + 100):
    """70/30 utterance-level split STRATIFIED so each speaker has utts in both
    train and test (otherwise classifier can't be evaluated on held-out
    speakers' own utts)."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for spk, group in meta_subset.groupby("speaker_id"):
        idx = rng.permutation(group.index.to_numpy())
        cut = max(1, int(round(train_frac * len(idx))))
        train_idx.extend(idx[:cut].tolist())
        test_idx.extend(idx[cut:].tolist())
    return np.asarray(train_idx), np.asarray(test_idx)


def held_out_speaker_accuracy(X_holdout: np.ndarray, meta_holdout: pd.DataFrame
                                ) -> float:
    """Within-holdout-speakers 70/30 utterance split; train 62-class classifier;
    return test accuracy."""
    spk_ids = sorted(meta_holdout["speaker_id"].unique())
    spk2int = {s: i for i, s in enumerate(spk_ids)}
    y = np.asarray([spk2int[s] for s in meta_holdout["speaker_id"]])
    train_idx, test_idx = within_speaker_split(meta_holdout.reset_index(drop=True))
    Xtr, Xte = X_holdout[train_idx], X_holdout[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=RNG_SEED)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return float(accuracy_score(yte, pred))


def main():
    out = RESULTS_DIR / "confound_holdout_summary.csv"
    if out.exists():
        prior = pd.read_csv(out)
        rows = prior.to_dict(orient="records")
        done = {r["model"] for r in rows}
        print(f"[resume] {len(rows)} prior rows; skipping: {sorted(done)}", flush=True)
    else:
        rows, done = [], set()
    for model in MODEL_ALIAS:
        if model in done:
            continue
        print(f"\n=== {model} ===", flush=True)
        Xws, meta = get_superb_representation(model)

        # 60/40 speaker split (matches confound_analysis.py)
        train_mask, test_mask = speaker_split(meta, train_frac=0.6)
        Xtr, Xte = Xws[train_mask], Xws[test_mask]
        meta_te = meta.loc[test_mask].reset_index(drop=True)

        speakers = sorted(meta["speaker_id"].unique())
        spk2int = {s: i for i, s in enumerate(speakers)}
        y_spk_tr = np.asarray([spk2int[s] for s in meta.loc[train_mask, "speaker_id"]])

        # Pre-INLP held-out speaker accuracy
        acc_pre = held_out_speaker_accuracy(Xte, meta_te)
        print(f"  pre-INLP held-out speaker acc = {acc_pre:.4f}", flush=True)

        # Run INLP (same as confound_analysis.py) on Xtr; project Xte alongside
        Wlist = []
        Xspk_tr_cur, Xspk_te_cur = Xtr.copy(), Xte.copy()
        chance = 1.0 / len(speakers) + 0.05
        for it in range(12):
            sc = StandardScaler().fit(Xspk_tr_cur)
            clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30,
                                tol=None, random_state=RNG_SEED, n_jobs=-1)
            clf.fit(sc.transform(Xspk_tr_cur), y_spk_tr)
            train_acc = accuracy_score(y_spk_tr, clf.predict(sc.transform(Xspk_tr_cur)))
            print(f"  [INLP] iter {it}: train spk acc = {train_acc:.4f}",
                  flush=True)
            if train_acc <= chance:
                break
            W = clf.coef_
            Wlist.append(W)
            Xspk_tr_cur = project_null_space(Xspk_tr_cur, W)
            Xspk_te_cur = project_null_space(Xspk_te_cur, W)

        # Post-INLP held-out speaker accuracy
        acc_post = held_out_speaker_accuracy(Xspk_te_cur, meta_te)
        print(f"  post-INLP held-out speaker acc = {acc_post:.4f}", flush=True)
        print(f"  drop = {acc_pre - acc_post:+.4f}  ({len(Wlist)} INLP iters)",
              flush=True)

        rows.append({
            "model": model,
            "n_train_speakers": int(meta.loc[train_mask, "speaker_id"].nunique()),
            "n_holdout_speakers": int(meta.loc[test_mask, "speaker_id"].nunique()),
            "holdout_pre_inlp_speaker_acc": acc_pre,
            "holdout_post_inlp_speaker_acc": acc_post,
            "inlp_iters": len(Wlist),
        })
        # Incremental save so a timeout doesn't lose results.
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [saved partial] {out}", flush=True)

    df = pd.read_csv(out)
    print(f"\nFinal results in {out}", flush=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
