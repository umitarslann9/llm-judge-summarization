"""Stage 1b - build the fixed, seeded evaluation subset.

Flattens SummEval (one row per article, each with N machine summaries + aligned
human scores) into one record per (article, summary) and writes the frozen
subset to data/subset.jsonl. Selection is seeded, so re-running reproduces the
identical file.

Output record schema:
    {id, article_id, summary_id, source, summary, references,
     human: {coherence, consistency, fluency, relevance}}
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (HUMAN_DIMS, load_config, project_paths, read_jsonl,
                    write_jsonl)

# Column-name candidates in mteb/summeval (kept flexible in case of variants).
SRC_KEYS = ["text", "source", "document", "article"]
MACHINE_KEYS = ["machine_summaries", "machine_summary", "summaries"]
REF_KEYS = ["human_summaries", "references", "reference_summaries"]


def _pick(row: dict, candidates: list[str]) -> str:
    for c in candidates:
        if c in row:
            return c
    raise KeyError(f"none of {candidates} found in columns {list(row.keys())}")


def _as_float(x) -> float:
    """Coerce a score cell to a float (averaging if it is a per-annotator list)."""
    if isinstance(x, (list, tuple)):
        vals = [float(v) for v in x if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")
    return float(x)


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    sub = cfg["subset"]
    rng = random.Random(sub["seed"])

    raw_path = paths["data_dir"] / "summeval_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} missing - run data/download.py first")
    rows = read_jsonl(raw_path)
    print(f"[preprocess] {len(rows)} articles loaded")

    src_key = _pick(rows[0], SRC_KEYS)
    machine_key = _pick(rows[0], MACHINE_KEYS)
    ref_key = _pick(rows[0], REF_KEYS)
    for d in HUMAN_DIMS:
        if d not in rows[0]:
            raise KeyError(f"human dimension '{d}' not found in columns "
                           f"{list(rows[0].keys())}")
    print(f"[preprocess] source='{src_key}' machine='{machine_key}' "
          f"refs='{ref_key}' dims={HUMAN_DIMS}")

    # Seeded selection of articles.
    n_art = min(sub["n_articles"], len(rows))
    art_idx = sorted(rng.sample(range(len(rows)), n_art))

    records = []
    for a in art_idx:
        row = rows[a]
        summaries = row[machine_key]
        n_sum = min(sub["summaries_per_article"], len(summaries))
        sum_idx = sorted(rng.sample(range(len(summaries)), n_sum))
        refs = row[ref_key] if isinstance(row[ref_key], list) else [row[ref_key]]
        for s in sum_idx:
            human = {d: _as_float(row[d][s]) for d in HUMAN_DIMS}
            records.append({
                "id": f"art{a:03d}_sum{s:02d}",
                "article_id": a,
                "summary_id": s,
                "source": row[src_key],
                "summary": summaries[s],
                "references": refs,
                "human": human,
            })

    write_jsonl(paths["subset"], records)
    print(f"[preprocess] wrote {len(records)} (article,summary) records "
          f"from {n_art} articles -> {paths['subset']}")


if __name__ == "__main__":
    main()
