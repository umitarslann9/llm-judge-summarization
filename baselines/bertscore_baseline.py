"""Baseline 2 - BERTScore.

Embedding-based semantic similarity between each machine summary and the human
reference summaries (multi-reference: BERTScore takes the best-matching ref).
Runs on CPU by default to avoid competing with Ollama for the 4 GB GPU.
Output: results/bertscore.jsonl, one row per (article, summary).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert_score import score as bertscore

from common import load_config, project_paths, read_jsonl, write_jsonl


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    bs_cfg = cfg.get("bertscore", {})
    records = read_jsonl(paths["subset"])

    cands = [r["summary"] for r in records]
    # Multi-reference: pass a list of reference strings per candidate.
    refs = [r["references"] for r in records]

    print(f"[bertscore] scoring {len(cands)} summaries on "
          f"{bs_cfg.get('device', 'cpu')} (model={bs_cfg.get('model_type', 'roberta-large')}) ...")
    P, R, F = bertscore(
        cands, refs,
        model_type=bs_cfg.get("model_type", "roberta-large"),
        lang=bs_cfg.get("lang", "en"),
        device=bs_cfg.get("device", "cpu"),
        verbose=True,
    )

    out = []
    for i, rec in enumerate(records):
        out.append({
            "id": rec["id"],
            "article_id": rec["article_id"],
            "summary_id": rec["summary_id"],
            "metric": "bertscore",
            "scores": {
                "bertscore_p": round(float(P[i]), 6),
                "bertscore_r": round(float(R[i]), 6),
                "bertscore_f": round(float(F[i]), 6),
            },
            "human": rec["human"],
        })

    out_path = paths["results_dir"] / "bertscore.jsonl"
    write_jsonl(out_path, out)
    print(f"[bertscore] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
