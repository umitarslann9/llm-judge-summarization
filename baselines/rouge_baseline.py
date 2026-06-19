"""Baseline 1 - ROUGE (ROUGE-1 / ROUGE-2 / ROUGE-L).

Classical n-gram overlap between each machine summary and the human reference
summaries. With multiple references we report the max F-measure over references
(standard practice). Output: results/rouge.jsonl, one row per (article, summary).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rouge_score import rouge_scorer
from tqdm import tqdm

from common import load_config, project_paths, read_jsonl, write_jsonl

METRICS = ["rouge1", "rouge2", "rougeL"]


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    records = read_jsonl(paths["subset"])

    scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=True)
    out = []
    for rec in tqdm(records, desc="rouge"):
        best = {m: 0.0 for m in METRICS}
        for ref in rec["references"]:
            s = scorer.score(ref, rec["summary"])
            for m in METRICS:
                best[m] = max(best[m], s[m].fmeasure)
        out.append({
            "id": rec["id"],
            "article_id": rec["article_id"],
            "summary_id": rec["summary_id"],
            "metric": "rouge",
            "scores": {m: round(best[m], 6) for m in METRICS},
            "human": rec["human"],
        })

    out_path = paths["results_dir"] / "rouge.jsonl"
    write_jsonl(out_path, out)
    print(f"[rouge] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
