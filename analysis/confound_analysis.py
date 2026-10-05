"""RQ4: Confound-controlled probing.

For each model, on the pooled 4-class cross-lingual data:
1. Compute the SUPERB-weighted-sum representation X (N, D).
2. Train linear probes for FOUR attributes on X:
     - emotion (4 classes)
     - speaker (many classes)
     - language (4 classes)
     - dataset (5 classes)
   Report each probe's accuracy. This shows what the embedding actually encodes.
3. Apply Iterative Null-space Projection (INLP, Ravfogel et al. 2020) to remove
   speaker information from X.
4. Re-train emotion probe on the cleaned X. Report accuracy.

If emotion accuracy survives speaker removal, the original finding is genuine.
If emotion accuracy collapses, much of the apparent "emotion" signal was speaker
identity. The latter would weaken the paper's claims; the former defends them.

INLP details: at each iteration, train a linear classifier for the unwanted
attribute (speaker), then project the embedding onto the null space of the
classifier's weight matrix. Repeat until the classifier reaches near-chance.
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]
RNG_SEED = 20260505


def speaker_split(meta: pd.DataFrame, train_frac: float = 0.6,
                  rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    speakers = sorted(meta["speaker_id"].unique())
    rng.shuffle(speakers)
    n_train = max(1, int(round(train_frac * len(speakers))))
    train_speakers = set(speakers[:n_train])
    train_mask = meta["speaker_id"].isin(train_speakers).to_numpy()
    return train_mask, ~train_mask


def get_superb_representation(model: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Pool per-layer embeddings, train the global SUPERB emotion probe, and return
    its weighted-sum representation alongside metadata.

    Nested / leakage-free: the layer weights are learned only on the 60% train-speaker
    split (train_frac=0.6, same seed as the downstream INLP/emotion split), so the 62
    held-out speakers never influence the representation the held-out emotion probe is
    evaluated on. (Previously the weights were learned on an 80% split of all speakers,
    which overlapped part of the held-out set; a leakage-free nested split was
    requested in review, so we tie the representation split to the analysis split.)"""
    embs, metas = [], []
    for d in DATASETS:
        e, m = load_embeddings(d, model, layer=None)
        m["language"] = DATASET_LANG[d]
        embs.append(e)
        metas.append(m)
    X = np.concatenate(embs, axis=0)  # (N, L, D)
    meta = pd.concat(metas, ignore_index=True)

    label_to_int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    train_mask, _ = speaker_split(meta, train_frac=0.6)
    Xtr = X[train_mask]
    ytr = np.asarray([label_to_int[y] for y in meta.loc[train_mask, "emotion"]])

    probe = SuperbProbe(n_layers=X.shape[1], dim=X.shape[2],
                       n_classes=len(EMOTION_4CLASS),
                       lr=1e-3, weight_decay=1e-4, epochs=100, batch_size=256,
                       seed=RNG_SEED)
    probe.fit(Xtr, ytr)
    Xws = probe.weighted_sum(X)  # (N, D)
    return Xws, meta


def linear_probe_acc(Xtr, ytr, Xte, yte) -> float:
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=RNG_SEED)
    clf.fit(sc.transform(Xtr), ytr)
    return accuracy_score(yte, clf.predict(sc.transform(Xte)))


def get_classifier_weights(Xtr, ytr) -> np.ndarray:
    """Train a linear classifier and return its coefficient matrix (n_classes, D)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=RNG_SEED)
    clf.fit(sc.transform(Xtr), ytr)
    # standardization absorbed; returned coef is in standardized space, but for
    # null-space projection we want the direction in the original feature space.
    # Approximation: scale weights back via the std (treating mean shift as ok).
    W = clf.coef_  # (n_classes, D)
    return W


def project_null_space(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project X onto the null space of W. X: (N, D), W: (k, D).
    Returns X_clean: (N, D) such that X_clean @ W^T ≈ 0."""
    # Compute the orthonormal basis of W's row space, then project X onto its null.
    # Using SVD: W = U Σ V^T, the null space is the columns of V corresponding to
    # zero singular values. Equivalently, projection = X - X @ V V^T where V is
    # the row space basis.
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    # rank: number of non-trivial singular values
    rank = (S > 1e-6).sum()
    Vrow = Vt[:rank]  # (rank, D), rows span W's row space
    projection = X @ Vrow.T @ Vrow  # X projected onto row space
    return X - projection            # remainder = projection onto null space


def inlp(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
         max_iter: int = 30, target_acc: float = 0.05) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Apply INLP to remove an attribute from the embedding.

    Iteratively: train classifier for the target attribute, project null-space
    until classifier accuracy reaches ~chance.
    Returns: (X_train_clean, X_test_clean, history of train_accs).
    """
    Xtr_cur, Xte_cur = Xtr.copy(), Xte.copy()
    history = []
    n_classes = len(np.unique(ytr))
    chance = max(target_acc, 1.0 / n_classes + 0.05)  # tolerate a bit above chance

    for i in range(max_iter):
        # Train classifier
        sc = StandardScaler().fit(Xtr_cur)
        clf = LogisticRegression(C=1.0, max_iter=300, random_state=RNG_SEED)
        clf.fit(sc.transform(Xtr_cur), ytr)
        train_acc = accuracy_score(ytr, clf.predict(sc.transform(Xtr_cur)))
        history.append(train_acc)
        if train_acc <= chance:
            break
        # Project away the classifier's directions
        W = clf.coef_
        Xtr_cur = project_null_space(Xtr_cur, W)
        Xte_cur = project_null_space(Xte_cur, W)
    return Xtr_cur, Xte_cur, history


def confound_analysis_for_model(model: str) -> dict:
    print(f"  [setup] loading + SUPERB", flush=True)
    Xws, meta = get_superb_representation(model)
    train_mask, test_mask = speaker_split(meta, train_frac=0.6)

    # Encode attribute labels (only what we need)
    emo2int = {e: i for i, e in enumerate(EMOTION_4CLASS)}
    speakers = sorted(meta["speaker_id"].unique())
    spk2int = {s: i for i, s in enumerate(speakers)}
    langs = sorted(meta["language"].unique())
    lang2int = {l: i for i, l in enumerate(langs)}
    datasets = sorted(meta["dataset"].unique())
    ds2int = {d: i for i, d in enumerate(datasets)}

    Xtr, Xte = Xws[train_mask], Xws[test_mask]
    y_emo_tr = np.asarray([emo2int[y] for y in meta.loc[train_mask, "emotion"]])
    y_emo_te = np.asarray([emo2int[y] for y in meta.loc[test_mask, "emotion"]])

    # For non-emotion attribute probes, use a within-speaker stratified split.
    # Use the SAME train indices as speaker-independent for emotion to keep it apples-to-apples.
    # For speaker probe diagnostic specifically, we need within-speaker, but with 199 classes
    # the LogisticRegression is slow. Subsample to ~10 speakers for the speaker probe sanity check.
    rng = np.random.default_rng(RNG_SEED)

    # Language probe (small label set, fast)
    Xall_idx = np.arange(len(meta))
    perm = rng.permutation(Xall_idx)
    cut = int(0.6 * len(meta))
    sl_tr_idx, sl_te_idx = perm[:cut], perm[cut:]
    y_lang_tr = np.asarray([lang2int[meta.iloc[i]["language"]] for i in sl_tr_idx])
    y_lang_te = np.asarray([lang2int[meta.iloc[i]["language"]] for i in sl_te_idx])
    y_ds_tr = np.asarray([ds2int[meta.iloc[i]["dataset"]] for i in sl_tr_idx])
    y_ds_te = np.asarray([ds2int[meta.iloc[i]["dataset"]] for i in sl_te_idx])

    print(f"  [diagnostic] emotion probe", flush=True)
    acc_emo = linear_probe_acc(Xtr, y_emo_tr, Xte, y_emo_te)
    print(f"  [diagnostic] emotion={acc_emo:.3f}", flush=True)

    print(f"  [diagnostic] language probe", flush=True)
    acc_lang = linear_probe_acc(Xws[sl_tr_idx], y_lang_tr, Xws[sl_te_idx], y_lang_te)
    print(f"  [diagnostic] language={acc_lang:.3f}", flush=True)

    print(f"  [diagnostic] dataset probe", flush=True)
    acc_ds = linear_probe_acc(Xws[sl_tr_idx], y_ds_tr, Xws[sl_te_idx], y_ds_te)
    print(f"  [diagnostic] dataset={acc_ds:.3f}", flush=True)

    # Step 3-4: INLP to remove speaker, then re-test emotion probe.
    # Subsample to a moderate-class speaker probe to keep INLP tractable.
    # Actually, the cleaner formulation: do INLP using 8-cluster speaker proxy
    # (group speakers into 8 buckets so the classifier is 8-class and tractable).
    # But the classical INLP is per-speaker. Compromise: use ALL speakers but with
    # liblinear solver (faster for high-class) and limited iterations.
    print(f"  [INLP] removing speaker subspace ({len(speakers)} speakers, max 12 iters, SGD solver)", flush=True)
    Wlist = []
    Xspk_tr_cur, Xspk_te_cur = Xtr.copy(), Xte.copy()
    y_spk_tr = np.asarray([spk2int[s] for s in meta.loc[train_mask, "speaker_id"]])
    chance = 1.0 / len(speakers) + 0.05
    for it in range(12):
        sc = StandardScaler().fit(Xspk_tr_cur)
        # SGDClassifier with log_loss is multinomial logistic regression via SGD.
        # ~5-10x faster than lbfgs LogisticRegression for 155 classes on 768D.
        clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=30, tol=None,
                             random_state=RNG_SEED, n_jobs=-1)
        clf.fit(sc.transform(Xspk_tr_cur), y_spk_tr)
        train_acc = accuracy_score(y_spk_tr, clf.predict(sc.transform(Xspk_tr_cur)))
        print(f"  [INLP] iter {it}: speaker train acc={train_acc:.3f}", flush=True)
        if train_acc <= chance:
            break
        W = clf.coef_
        Wlist.append(W)
        Xspk_tr_cur = project_null_space(Xspk_tr_cur, W)
        Xspk_te_cur = project_null_space(Xspk_te_cur, W)
    print(f"  [INLP] applied {len(Wlist)} projections", flush=True)

    # Apply same projections to the speaker-independent emotion train/test
    Xtr_clean = Xtr.copy()
    Xte_clean = Xte.copy()
    for W in Wlist:
        Xtr_clean = project_null_space(Xtr_clean, W)
        Xte_clean = project_null_space(Xte_clean, W)
    print(f"  [diagnostic] emotion probe AFTER speaker INLP", flush=True)
    acc_emo_after = linear_probe_acc(Xtr_clean, y_emo_tr, Xte_clean, y_emo_te)
    print(f"  [diagnostic] emotion after INLP={acc_emo_after:.3f}", flush=True)

    return {
        "model": model,
        "acc_emotion_baseline": acc_emo,
        "acc_language": acc_lang,
        "acc_dataset": acc_ds,
        "acc_emotion_after_speaker_inlp": acc_emo_after,
        "n_speakers": len(speakers),
        "inlp_iters": len(Wlist),
    }


def main():
    rows = []
    out = RESULTS_DIR / "confound_summary.csv"
    for m in MODEL_ALIAS:
        print(f"\n=== {m} ===", flush=True)
        res = confound_analysis_for_model(m)
        rows.append(res)
        print(f"  SUMMARY emotion baseline: {res['acc_emotion_baseline']:.3f}", flush=True)
        print(f"  SUMMARY language: {res['acc_language']:.3f}", flush=True)
        print(f"  SUMMARY dataset: {res['acc_dataset']:.3f}", flush=True)
        print(f"  SUMMARY emotion after speaker INLP: {res['acc_emotion_after_speaker_inlp']:.3f}  "
              f"(drop {res['acc_emotion_baseline'] - res['acc_emotion_after_speaker_inlp']:+.3f})", flush=True)
        # Incremental save so a timeout doesn't lose all results.
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [saved partial] {out}", flush=True)
    print(f"\nFinal results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
