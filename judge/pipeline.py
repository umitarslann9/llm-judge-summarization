"""Shared LLM-as-a-Judge loop: build prompt -> call model -> parse scores.

Runs the same rubric+CoT prompt through either the small local judge (Ollama)
or the large API judge (Gemini), selected with --judge. Writes one JSONL row per
(article, summary) with the four parsed scores plus token/latency metadata.

Usage:
    python judge/pipeline.py --judge small
    python judge/pipeline.py --judge large --limit 5      # smoke test first
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from common import (fill_prompt, load_config, load_env, project_paths,
                    read_jsonl, read_prompt, truncate_words, write_jsonl)
from judge.parse import parse_rubric_scores

MAX_PARSE_RETRIES = 2
RAW_KEEP_CHARS = 2000


def make_backend(judge: str, cfg: dict):
    if judge == "small":
        from judge.backend_local import build_backend
        return build_backend(cfg["small_judge"]), cfg["small_judge"]["model"]
    elif judge == "large":
        from judge.backend_api import build_backend
        return build_backend(cfg["large_judge"]), cfg["large_judge"]["model"]
    raise ValueError(f"unknown judge: {judge}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["small", "large"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N records (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="ignore any existing output and start fresh")
    args = ap.parse_args()

    load_env()
    cfg = load_config()
    paths = project_paths(cfg)
    max_words = cfg["subset"].get("max_source_words", 600)

    records = read_jsonl(paths["subset"])
    if args.limit:
        records = records[: args.limit]

    out_path = paths["results_dir"] / f"judge_{args.judge}.jsonl"
    done = {}
    if out_path.exists() and not args.overwrite:
        done = {r["id"]: r for r in read_jsonl(out_path)}
        print(f"[resume] {len(done)} records already scored in {out_path.name}")

    backend, model_name = make_backend(args.judge, cfg)
    template = read_prompt(paths["prompts_dir"], "rubric_cot.txt")

    results = [done[r["id"]] for r in records if r["id"] in done]
    todo = [r for r in records if r["id"] not in done]
    print(f"[{args.judge} judge] model={model_name}  to score: {len(todo)}")

    n_parse_fail = 0
    for rec in tqdm(todo, desc=f"{args.judge} judge"):
        prompt = fill_prompt(
            template,
            source=truncate_words(rec["source"], max_words),
            summary=rec["summary"],
        )
        scores, raw, last = None, "", None
        for _ in range(1 + MAX_PARSE_RETRIES):
            last = backend.chat(prompt)
            raw = last["text"]
            scores = parse_rubric_scores(raw)
            if scores is not None:
                break
        if scores is None:
            n_parse_fail += 1

        row = {
            "id": rec["id"],
            "article_id": rec["article_id"],
            "summary_id": rec["summary_id"],
            "judge": args.judge,
            "model": model_name,
            "scores": scores,
            "parse_ok": scores is not None,
            "raw": raw[:RAW_KEEP_CHARS],
            "prompt_tokens": last["prompt_tokens"],
            "completion_tokens": last["completion_tokens"],
            "total_tokens": last["total_tokens"],
            "latency_s": round(last["latency_s"], 4),
            "human": rec["human"],
        }
        results.append(row)
        # Write after every record so a long run can be resumed safely.
        write_jsonl(out_path, results)

    print(f"[done] {len(results)} rows -> {out_path}  (parse failures: {n_parse_fail})")


if __name__ == "__main__":
    main()
