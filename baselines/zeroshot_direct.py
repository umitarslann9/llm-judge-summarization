"""Baseline 3 - Zero-shot Direct Scoring.

The same small open model used for the small judge, but prompted to emit a single
overall quality score (1-5) with NO rubric and NO chain-of-thought. Isolates how
much of the small judge's performance comes from prompt design vs. raw ability.
Output: results/zeroshot.jsonl, one row per (article, summary).

Usage:
    python baselines/zeroshot_direct.py [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from common import (fill_prompt, load_config, load_env, project_paths,
                    read_jsonl, read_prompt, truncate_words, write_jsonl)
from judge.backend_local import build_backend
from judge.parse import parse_single_score

MAX_PARSE_RETRIES = 2
RAW_KEEP_CHARS = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_config()
    paths = project_paths(cfg)
    max_words = cfg["subset"].get("max_source_words", 600)

    records = read_jsonl(paths["subset"])
    if args.limit:
        records = records[: args.limit]

    out_path = paths["results_dir"] / "zeroshot.jsonl"
    done = {}
    if out_path.exists() and not args.overwrite:
        done = {r["id"]: r for r in read_jsonl(out_path)}
        print(f"[resume] {len(done)} already scored in {out_path.name}")

    backend = build_backend(cfg["zeroshot"])
    template = read_prompt(paths["prompts_dir"], "direct_score.txt")

    results = [done[r["id"]] for r in records if r["id"] in done]
    todo = [r for r in records if r["id"] not in done]
    print(f"[zeroshot] model={cfg['zeroshot']['model']}  to score: {len(todo)}")

    n_fail = 0
    for rec in tqdm(todo, desc="zeroshot"):
        prompt = fill_prompt(
            template,
            source=truncate_words(rec["source"], max_words),
            summary=rec["summary"],
        )
        score, raw, last = None, "", None
        for _ in range(1 + MAX_PARSE_RETRIES):
            last = backend.chat(prompt)
            raw = last["text"]
            score = parse_single_score(raw)
            if score is not None:
                break
        if score is None:
            n_fail += 1
        results.append({
            "id": rec["id"],
            "article_id": rec["article_id"],
            "summary_id": rec["summary_id"],
            "metric": "zeroshot",
            "score": score,
            "parse_ok": score is not None,
            "raw": raw[:RAW_KEEP_CHARS],
            "prompt_tokens": last["prompt_tokens"],
            "completion_tokens": last["completion_tokens"],
            "total_tokens": last["total_tokens"],
            "latency_s": round(last["latency_s"], 4),
            "human": rec["human"],
        })
        write_jsonl(out_path, results)

    print(f"[zeroshot] wrote {len(results)} rows -> {out_path}  (parse fails: {n_fail})")


if __name__ == "__main__":
    main()
