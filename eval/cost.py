"""Stage 5 - cost & speed per method.

Aggregates token usage and latency for the model-based methods (zero-shot, small
judge, large judge) from their result files. The large judge is billed per token
(Gemini free tier => $0 here, but tokens are reported so the trade-off is visible);
the small judge / zero-shot are local => no monetary cost, only latency / GPU time.
Output: results/cost.csv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from common import load_config, project_paths, read_jsonl

# label -> (filename, is_local)
SOURCES = {
    "Zero-shot (local)": ("zeroshot.jsonl", True),
    "Small judge (local)": ("judge_small.jsonl", True),
    "Large judge (Gemini)": ("judge_large.jsonl", False),
}

# Approximate API list price, USD per 1M tokens: (input, output).
PRICING = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "llama-3.3-70b": (0.59, 0.79),
}


def estimate_cost_usd(model: str, in_tok: int, out_tok: int):
    for key, (pin, pout) in PRICING.items():
        if key in (model or ""):
            return in_tok / 1e6 * pin + out_tok / 1e6 * pout
    return None


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)

    rows = []
    for label, (fname, is_local) in SOURCES.items():
        fpath = paths["results_dir"] / fname
        if not fpath.exists():
            print(f"[skip] {label}: {fname} not found")
            continue
        data = read_jsonl(fpath)
        n = len(data)
        lat = [r.get("latency_s", 0.0) for r in data]
        ptok = sum(r.get("prompt_tokens", 0) for r in data)
        ctok = sum(r.get("completion_tokens", 0) for r in data)
        ttok = sum(r.get("total_tokens", 0) for r in data)
        total_time = sum(lat)
        if is_local:
            monetary = "$0 (local GPU)"
        else:
            model = data[0].get("model", "") if data else ""
            usd = estimate_cost_usd(model, ptok, ctok)
            monetary = f"~${usd:.2f}" if usd is not None else "n/a"
        rows.append({
            "method": label,
            "n_summaries": n,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "total_tokens": ttok,
            "tokens_per_summary": round(ttok / n, 1) if n else 0,
            "mean_latency_s": round(total_time / n, 3) if n else 0,
            "total_time_s": round(total_time, 1),
            "throughput_summ_per_min": round(60 * n / total_time, 2) if total_time else 0,
            "monetary_cost": monetary,
        })

    df = pd.DataFrame(rows)
    out_csv = paths["results_dir"] / "cost.csv"
    df.to_csv(out_csv, index=False)
    print(f"[cost] wrote {out_csv}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
