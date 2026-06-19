"""Stage 5 - bootstrap confidence intervals on human correlation.

The evaluation subset is small, so point correlations are noisy. This estimates a
95% bootstrap CI for each method's average (over the four dimensions) Spearman
correlation with human scores, by resampling (article, summary) records with
replacement. Output: results/bootstrap.csv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import HUMAN_DIMS, load_config, project_paths, read_jsonl
from eval.correlation import METHODS

B_DEFAULT = 1000
SEED = 42


def avg_spearman(data, fn, idx) -> float:
    dim_vals = []
    for hdim in HUMAN_DIMS:
        xs, ys = [], []
        for i in idx:
            r = data[i]
            pred = fn(r, hdim)
            gold = r["human"].get(hdim)
            if pred is None or gold is None or (isinstance(gold, float) and math.isnan(gold)):
                continue
            xs.append(pred)
            ys.append(gold)
        if len(xs) >= 3 and len(set(xs)) > 1 and len(set(ys)) > 1:
            dim_vals.append(spearmanr(xs, ys).correlation)
    return float(np.mean(dim_vals)) if dim_vals else float("nan")


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    B = cfg.get("bootstrap", {}).get("n_resamples", B_DEFAULT)
    rng = np.random.default_rng(SEED)

    rows = []
    for method, (fname, fn) in METHODS.items():
        fpath = paths["results_dir"] / fname
        if not fpath.exists():
            print(f"[skip] {method}: {fname} not found")
            continue
        data = read_jsonl(fpath)
        n = len(data)
        point = avg_spearman(data, fn, range(n))
        samples = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            val = avg_spearman(data, fn, idx)
            if not math.isnan(val):
                samples.append(val)
        lo, hi = (np.percentile(samples, [2.5, 97.5]) if samples else (float("nan"),) * 2)
        rows.append({
            "method": method,
            "avg_spearman": round(point, 4),
            "ci95_low": round(float(lo), 4),
            "ci95_high": round(float(hi), 4),
            "n_resamples": len(samples),
        })
        print(f"  {method:14s} {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    df = pd.DataFrame(rows)
    out_csv = paths["results_dir"] / "bootstrap.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[bootstrap] wrote {out_csv}")


if __name__ == "__main__":
    main()
