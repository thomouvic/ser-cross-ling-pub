"""INLP rank-matched random-projection control.

Rationale: after INLP removes a speaker subspace of total rank r,
also project the representation onto the orthogonal complement of a RANDOM
subspace of the same rank r (averaged over seeds) and retrain the emotion probe.
This isolates how much of the post-INLP emotion drop is specific to removing
speaker-discriminative directions versus a generic capacity-reduction effect of
removing r out of 768 dimensions.

For each model we report, on the SUPERB-pooled cross-lingual representation:
  - total removed rank r (dimension of the accumulated INLP null-space);
  - residual speaker decodability:
      * train-speakers post-INLP (final INLP train-speaker accuracy);
      * held-out speakers pre-INLP and post-INLP (62-speaker, 70/30 within-split);
  - emotion accuracy under (i) speaker-INLP and (ii) rank-r random projection.

Split is leakage-free / nested: the speaker classifier (INLP) and the random
projection are both fit using ONLY the 93 train-speakers; emotion is evaluated on
the 62 held-out speakers. The random projection uses no labels at all, so it is
trivially leakage-free; it is applied identically to train and held-out data.

This mirrors confound_analysis.py / confound_holdout.py exactly for the INLP arm
(same SUPERB probe, same 60/40 speaker split, same SGD speaker classifier, same
projection operator), and adds the random-projection arm and the rank readout.

Outputs:
  results/confound_random_control.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load import MODEL_ALIAS
from confound_analysis import (
    get_superb_representation, speaker_split, project_null_space,
    linear_probe_acc, EMOTION_4CLASS, RNG_SEED,
)
from confound_holdout import held_out_speaker_accuracy

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
N_RANDOM_SEEDS = 5      # random subspaces to average the emotion control over
HOLDOUT_RAND_SEEDS = 2  # random subspaces for the slow 62-class holdout probe
SVD_TOL = 1e-6


def effective_rank(X: np.ndarray, rel_tol: float = 1e-6) -> int:
    """Numerical rank of X via SVD with a relative tolerance on singular values.
    For the post-INLP representation this counts the surviving dimensions; the
    difference from the ambient 768 is the number of dimensions the (non-orthogonal
    sequential) projection actually collapsed."""
    s = np.linalg.svd(X, compute_uv=False)
    return int((s > rel_tol * s.max()).sum())


def project_out_basis(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project X onto the orthogonal complement of the subspace spanned by the
    orthonormal rows of Q (same operator as project_null_space, for an already
    orthonormal basis)."""
    return X - X @ Q.T @ Q


def random_orthonormal(rank: int, dim: int, seed: int) -> np.ndarray:
    """Orthonormal basis (rank x dim) of a uniformly random rank-dim subspace."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((dim, rank))
    Q, _ = np.linalg.qr(G)     # (dim, rank), columns orthonormal
    return Q.T                  # (rank, dim), rows orthonormal


def run_model(model: str) -> dict:
    print(f"\n=== {model} ===", flush=True)
    Xws, meta = get_superb_representation(model)
    # Nested, leakage-free 60/40 speaker split (matches confound_analysis.py).
    train_mask, test_mask = speaker_split(meta, train_frac=0.6)
    Xtr, Xte = Xws[train_mask], Xws[test_mask]
    meta_te = meta.loc[test_mask].reset_index(drop=True)

    emo2int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    y_emo_tr = np.asarray([emo2int[y] for y in meta.loc[train_mask, "emotion"]])
    y_emo_te = np.asarray([emo2int[y] for y in meta.loc[test_mask, "emotion"]])

    speakers = sorted(meta["speaker_id"].unique())
    spk2int = {s: i for i, s in enumerate(speakers)}
    y_spk_tr = np.asarray([spk2int[s] for s in meta.loc[train_mask, "speaker_id"]])
    chance = 1.0 / len(speakers) + 0.05

    # Emotion baseline (pre-INLP), 93-train / 62-holdout speaker-independent.
    acc_emo_pre = linear_probe_acc(Xtr, y_emo_tr, Xte, y_emo_te)
    print(f"  emotion pre = {acc_emo_pre:.4f}", flush=True)

    # Held-out speaker decodability pre-INLP.
    holdout_pre = held_out_speaker_accuracy(Xte, meta_te)
    print(f"  holdout speaker pre = {holdout_pre:.4f}", flush=True)

    dim = Xtr.shape[1]
    rank_pre = effective_rank(Xtr)
    print(f"  Xtr.shape = {Xtr.shape}; pre-INLP effective rank = {rank_pre}", flush=True)

    # --- INLP arm (identical projection sequence to confound_analysis.py) ---
    Wlist = []
    Xspk_tr_cur, Xspk_te_cur = Xtr.copy(), Xte.copy()
    for it in range(12):
        sc = StandardScaler().fit(Xspk_tr_cur)
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=None,
                            random_state=RNG_SEED, n_jobs=-1)
        clf.fit(sc.transform(Xspk_tr_cur), y_spk_tr)
        train_acc = accuracy_score(y_spk_tr, clf.predict(sc.transform(Xspk_tr_cur)))
        print(f"  [INLP] iter {it}: train spk acc = {train_acc:.4f}", flush=True)
        if train_acc <= chance:
            break
        W = clf.coef_
        Wlist.append(W)
        Xspk_tr_cur = project_null_space(Xspk_tr_cur, W)
        Xspk_te_cur = project_null_space(Xspk_te_cur, W)

    # Post-INLP train-speaker decodability (re-fit on the projected train data).
    sc = StandardScaler().fit(Xspk_tr_cur)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=None,
                        random_state=RNG_SEED, n_jobs=-1)
    clf.fit(sc.transform(Xspk_tr_cur), y_spk_tr)
    spk_train_post = accuracy_score(y_spk_tr, clf.predict(sc.transform(Xspk_tr_cur)))

    # Emotion after speaker-INLP (project emotion train/test through Wlist).
    Xtr_inlp, Xte_inlp = Xtr.copy(), Xte.copy()
    for W in Wlist:
        Xtr_inlp = project_null_space(Xtr_inlp, W)
        Xte_inlp = project_null_space(Xte_inlp, W)
    acc_emo_inlp = linear_probe_acc(Xtr_inlp, y_emo_tr, Xte_inlp, y_emo_te)

    # Total removed rank r = dimensions the sequential projection actually
    # collapsed = pre-INLP rank minus surviving rank of the cleaned train rep.
    # (The raw union of the per-iteration speaker row-spaces spans all 768 dims
    # because the iterations are non-orthogonal under the standardization step, so
    # the honest "removed rank" is this net rank deficiency, not that union.)
    rank_post = effective_rank(Xtr_inlp)
    r = int(rank_pre - rank_post)
    print(f"  [INLP] applied {len(Wlist)} projections; post-INLP rank = {rank_post}; "
          f"removed rank r = {r}", flush=True)
    print(f"  emotion post-INLP = {acc_emo_inlp:.4f}", flush=True)

    # Held-out speaker decodability post-INLP.
    holdout_post_inlp = held_out_speaker_accuracy(Xspk_te_cur, meta_te)
    print(f"  holdout speaker post-INLP = {holdout_post_inlp:.4f}", flush=True)

    # --- Rank-r random-projection control (averaged over seeds) ---
    # Emotion (fast 4-class probe) averaged over all seeds; the held-out speaker
    # decodability (slow 62-class probe) barely varies across random subspaces, so
    # we average it over the first HOLDOUT_RAND_SEEDS seeds only.
    rand_emo, rand_holdout_post, resid_fracs = [], [], []
    for s in range(N_RANDOM_SEEDS):
        Q = random_orthonormal(r, dim, seed=RNG_SEED + 7919 * (s + 1))
        Xtr_rp = project_out_basis(Xtr, Q)
        Xte_rp = project_out_basis(Xte, Q)
        resid_fracs.append(float(np.linalg.norm(Xtr_rp) / np.linalg.norm(Xtr)))
        rand_emo.append(linear_probe_acc(Xtr_rp, y_emo_tr, Xte_rp, y_emo_te))
        if s < HOLDOUT_RAND_SEEDS:
            rand_holdout_post.append(held_out_speaker_accuracy(Xte_rp, meta_te))
    acc_emo_rand = float(np.mean(rand_emo))
    acc_emo_rand_std = float(np.std(rand_emo))
    holdout_post_rand = float(np.mean(rand_holdout_post))
    print(f"  [random] rank {r}: residual-norm frac (train) mean = "
          f"{np.mean(resid_fracs):.3f}", flush=True)
    print(f"  emotion post-random(rank {r}) = {acc_emo_rand:.4f} +- {acc_emo_rand_std:.4f}",
          flush=True)
    print(f"  holdout speaker post-random = {holdout_post_rand:.4f}", flush=True)

    return {
        "model": model,
        "removed_rank_r": r,
        "rank_pre": int(rank_pre),
        "rank_post": int(rank_post),
        "inlp_iters": len(Wlist),
        "emo_pre": acc_emo_pre,
        "emo_post_inlp": acc_emo_inlp,
        "emo_post_random_mean": acc_emo_rand,
        "emo_post_random_std": acc_emo_rand_std,
        "spk_train_post_inlp": float(spk_train_post),
        "holdout_spk_pre": float(holdout_pre),
        "holdout_spk_post_inlp": float(holdout_post_inlp),
        "holdout_spk_post_random": holdout_post_rand,
        "n_random_seeds": N_RANDOM_SEEDS,
    }


def main():
    out = RESULTS_DIR / "confound_random_control.csv"
    if out.exists():
        prior = pd.read_csv(out)
        rows = prior.to_dict(orient="records")
        done = {r["model"] for r in rows}
        print(f"[resume] {len(rows)} prior rows; skipping {sorted(done)}", flush=True)
    else:
        rows, done = [], set()
    for m in MODEL_ALIAS:
        if m in done:
            continue
        rows.append(run_model(m))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [saved partial] {out}", flush=True)
    df = pd.read_csv(out)
    print(f"\nFinal results in {out}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
