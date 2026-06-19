"""Stage 5 - judge-vs-judge agreement (large vs. small).

How closely does the small judge track the large judge, independent of human
scores? Reports, per judge dimension and overall: Pearson correlation, mean score
difference (small - large, signals strictness bias) and mean absolute difference.
Output: results/agreement.csv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.stats import pearsonr

from common import JUDGE_DIMS, load_config, project_paths, read_jsonl


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)

    small_path = paths["results_dir"] / "judge_small.jsonl"
    large_path = paths["results_dir"] / "judge_large.jsonl"
    for p in (small_path, large_path):
        if not p.exists():
            print(f"[agreement] missing {p.name} - run both judges first")
            return

    small = {r["id"]: r for r in read_jsonl(small_path) if r.get("parse_ok")}
    large = {r["id"]: r for r in read_jsonl(large_path) if r.get("parse_ok")}
    ids = sorted(set(small) & set(large))
    print(f"[agreement] {len(ids)} summaries scored by both judges")

    rows = []
    all_s, all_l = [], []
    for dim in JUDGE_DIMS:
        s = [small[i]["scores"][dim] for i in ids]
        l = [large[i]["scores"][dim] for i in ids]
        all_s += s
        all_l += l
        pr = pearsonr(s, l)[0] if len(set(s)) > 1 and len(set(l)) > 1 else float("nan")
        mean_diff = sum(a - b for a, b in zip(s, l)) / len(s)
        mae = sum(abs(a - b) for a, b in zip(s, l)) / len(s)
        rows.append({"dimension": dim, "n": len(ids), "pearson": pr,
                     "mean_diff_small_minus_large": mean_diff, "mean_abs_diff": mae})

    overall_pr = pearsonr(all_s, all_l)[0] if len(set(all_s)) > 1 else float("nan")
    rows.append({
        "dimension": "overall", "n": len(all_s), "pearson": overall_pr,
        "mean_diff_small_minus_large": sum(a - b for a, b in zip(all_s, all_l)) / len(all_s),
        "mean_abs_diff": sum(abs(a - b) for a, b in zip(all_s, all_l)) / len(all_s),
    })

    df = pd.DataFrame(rows)
    out_csv = paths["results_dir"] / "agreement.csv"
    df.to_csv(out_csv, index=False)
    print(f"[agreement] wrote {out_csv}\n")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
