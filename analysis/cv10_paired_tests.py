"""Per-fold paired tests for the CV10 side-table (Table III).

Reads results/probe_superb_full_cv10_summary.csv which has per-fold accuracies
in semicolon-separated form, computes paired t-tests across the 10 folds for
each (model_a, model_b, dataset) comparison.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_per_fold(s: str) -> np.ndarray:
    return np.asarray([float(x) for x in s.split(";")])


def main():
    df = pd.read_csv(RESULTS_DIR / "probe_superb_full_cv10_summary.csv")
    rows = []
    for d in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == d]
        models = sub["model"].tolist()
        per_fold = {m: parse_per_fold(sub[sub["model"] == m]["acc_per_fold"].iloc[0]) for m in models}
        for a in ["hubert", "wavlm"]:
            for b in ["emotion2vec"]:
                if a not in per_fold or b not in per_fold:
                    continue
                acc_a = per_fold[a]; acc_b = per_fold[b]
                t_stat, t_p = stats.ttest_rel(acc_a, acc_b)
                rows.append({
                    "dataset": d, "model_a": a, "model_b": b,
                    "n_folds": len(acc_a),
                    "mean_a": acc_a.mean(), "mean_b": acc_b.mean(),
                    "mean_diff": (acc_a - acc_b).mean(),
                    "t_pvalue": float(t_p),
                })
        # Also HuBERT vs WavLM for completeness
        if "hubert" in per_fold and "wavlm" in per_fold:
            acc_a, acc_b = per_fold["hubert"], per_fold["wavlm"]
            t_stat, t_p = stats.ttest_rel(acc_a, acc_b)
            rows.append({
                "dataset": d, "model_a": "hubert", "model_b": "wavlm",
                "n_folds": len(acc_a),
                "mean_a": acc_a.mean(), "mean_b": acc_b.mean(),
                "mean_diff": (acc_a - acc_b).mean(),
                "t_pvalue": float(t_p),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cv10_paired_tests.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
