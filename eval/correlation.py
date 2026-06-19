"""Stage 5 - human-alignment correlations.

For every method, correlate its scores against the SummEval human ratings on each
of the four dimensions, plus the 4-dimension average. The two LLM judges use their
matched dimension per human dimension (faithfulness<->consistency, and the three
same-named dims); the single-value methods (ROUGE, BERTScore, zero-shot) use their
one score against each human dimension, as is standard for reference metrics.

Reports Spearman, Kendall-tau and Pearson. Output: results/correlation.csv (tidy)
and a printed summary table of Spearman correlations.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from common import HUMAN_DIMS, JUDGE_TO_HUMAN, load_config, project_paths, read_jsonl

HUMAN_TO_JUDGE = {v: k for k, v in JUDGE_TO_HUMAN.items()}


def judge_pred(row, human_dim):
    sc = row.get("scores")
    if not sc:
        return None
    return sc.get(HUMAN_TO_JUDGE[human_dim])


# method label -> (results filename, predictor function(row, human_dim))
METHODS = {
    "ROUGE-1": ("rouge.jsonl", lambda r, h: r["scores"]["rouge1"]),
    "ROUGE-2": ("rouge.jsonl", lambda r, h: r["scores"]["rouge2"]),
    "ROUGE-L": ("rouge.jsonl", lambda r, h: r["scores"]["rougeL"]),
    "BERTScore-F": ("bertscore.jsonl", lambda r, h: r["scores"]["bertscore_f"]),
    "Zero-shot": ("zeroshot.jsonl", lambda r, h: r.get("score")),
    "Small judge": ("judge_small.jsonl", judge_pred),
    "Large judge": ("judge_large.jsonl", judge_pred),
}


def _corr(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan"), float("nan"), float("nan")
    sp = spearmanr(xs, ys).correlation
    kt = kendalltau(xs, ys).correlation
    pr = pearsonr(xs, ys)[0]
    return sp, kt, pr


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)

    rows = []
    for method, (fname, fn) in METHODS.items():
        fpath = paths["results_dir"] / fname
        if not fpath.exists():
            print(f"[skip] {method}: {fname} not found")
            continue
        data = read_jsonl(fpath)
        dim_sp = []
        for hdim in HUMAN_DIMS:
            xs, ys = [], []
            for r in data:
                pred = fn(r, hdim)
                gold = r["human"].get(hdim)
                if pred is None or gold is None or (isinstance(gold, float) and math.isnan(gold)):
                    continue
                xs.append(pred)
                ys.append(gold)
            sp, kt, pr = _corr(xs, ys)
            dim_sp.append(sp)
            rows.append({"method": method, "dimension": hdim, "n": len(xs),
                         "spearman": sp, "kendall": kt, "pearson": pr})
        valid = [s for s in dim_sp if not math.isnan(s)]
        avg = sum(valid) / len(valid) if valid else float("nan")
        rows.append({"method": method, "dimension": "average", "n": len(data),
                     "spearman": avg, "kendall": float("nan"), "pearson": float("nan")})

    df = pd.DataFrame(rows)
    out_csv = paths["results_dir"] / "correlation.csv"
    df.to_csv(out_csv, index=False)
    print(f"[correlation] wrote {out_csv}\n")

    # Pretty Spearman pivot (methods x dimensions).
    pivot = df.pivot(index="method", columns="dimension", values="spearman")
    cols = HUMAN_DIMS + ["average"]
    pivot = pivot.reindex(columns=[c for c in cols if c in pivot.columns])
    order = [m for m in METHODS if m in pivot.index]
    pivot = pivot.reindex(order)
    print("Spearman correlation with human scores:")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
