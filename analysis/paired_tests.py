"""Paired tests across 5 seeds for the headline accuracy gaps.

For each headline experiment (RQ1, Claim 1 SI, Claim 2 Thai), compute paired
t-tests and Wilcoxon signed-rank tests across the 5 matched seeds for each
(modelA vs modelB, dataset) comparison.

Replaces the hand-wavy "gap is larger than training-noise std" claim with
proper paired statistical inference.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def paired_tests_for(experiment: str, df: pd.DataFrame, model_a: str, model_b: str):
    """For each dataset, paired t-test and Wilcoxon between model_a and model_b
    across the 5 seeds."""
    rows = []
    sub = df[df["experiment"] == experiment]
    for d in sorted(sub["dataset"].unique()):
        a = sub[(sub["model"] == model_a) & (sub["dataset"] == d)].sort_values("seed")
        b = sub[(sub["model"] == model_b) & (sub["dataset"] == d)].sort_values("seed")
        seeds_a = a["seed"].tolist()
        seeds_b = b["seed"].tolist()
        common = sorted(set(seeds_a) & set(seeds_b))
        if len(common) < 2:
            continue
        a_acc = a[a["seed"].isin(common)].sort_values("seed")["accuracy"].to_numpy()
        b_acc = b[b["seed"].isin(common)].sort_values("seed")["accuracy"].to_numpy()
        diff = a_acc - b_acc  # positive means model_a > model_b
        if np.allclose(diff, 0):
            t_p = 1.0; w_p = 1.0
        else:
            t_stat, t_p = stats.ttest_rel(a_acc, b_acc)
            try:
                w_stat, w_p = stats.wilcoxon(a_acc, b_acc, alternative="two-sided")
            except ValueError:
                w_p = float("nan")
        rows.append({
            "experiment": experiment, "dataset": d,
            "model_a": model_a, "model_b": model_b,
            "n_seeds": len(common),
            "mean_a": float(a_acc.mean()), "mean_b": float(b_acc.mean()),
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "t_pvalue": float(t_p), "wilcoxon_pvalue": float(w_p),
        })
    return rows


def main():
    df = pd.read_csv(RESULTS_DIR / "multi_seed_per_seed.csv")
    print(f"Loaded {len(df)} per-seed rows. Experiments: {sorted(df['experiment'].unique())}")
    print()

    all_rows = []

    # RQ1: 3 models, all pairwise comparisons we care about
    for a, b in [("hubert", "emotion2vec"), ("wavlm", "emotion2vec"), ("hubert", "wavlm")]:
        all_rows += paired_tests_for("rq1", df, a, b)

    # Claim 1 SI: same comparisons
    for a, b in [("hubert", "emotion2vec"), ("wavlm", "emotion2vec"), ("hubert", "wavlm")]:
        all_rows += paired_tests_for("claim1_si", df, a, b)

    # Claim 2 Thai: 4 models, focus on _plus_base vs others
    for a, b in [("hubert", "emotion2vec_plus_base"),
                  ("wavlm", "emotion2vec_plus_base"),
                  ("emotion2vec_base", "emotion2vec_plus_base"),
                  ("hubert", "wavlm"),
                  ("hubert", "emotion2vec_base"),
                  ("wavlm", "emotion2vec_base")]:
        all_rows += paired_tests_for("claim2_thai", df, a, b)

    out = pd.DataFrame(all_rows)
    out_csv = RESULTS_DIR / "paired_tests.csv"
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
