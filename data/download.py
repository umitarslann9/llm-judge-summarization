"""Stage 1a - download SummEval from the Hugging Face Hub.

Pulls the configured dataset (default: mteb/summeval), prints its schema so the
preprocessing step can be verified against it, and dumps the raw rows to
data/summeval_raw.jsonl for inspection/reuse.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from common import ensure_dir, load_config, project_paths


def main() -> None:
    cfg = load_config()
    paths = project_paths(cfg)
    name = cfg["subset"]["dataset"]
    want_split = cfg["subset"].get("split", "test")

    print(f"[download] loading {name} ...")
    ds = load_dataset(name)
    splits = list(ds.keys())
    split = want_split if want_split in splits else splits[0]
    print(f"[download] available splits: {splits} -> using '{split}'")

    data = ds[split]
    print(f"[download] {len(data)} rows")
    print(f"[download] columns: {data.column_names}")

    sample = data[0]
    print("[download] sample row field types/lengths:")
    for k, v in sample.items():
        if isinstance(v, list):
            print(f"  - {k}: list(len={len(v)}), first={repr(v[0])[:80]}")
        else:
            print(f"  - {k}: {type(v).__name__} = {repr(v)[:80]}")

    out = ensure_dir(paths["data_dir"]) / "summeval_raw.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for row in data:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[download] wrote raw rows -> {out}")


if __name__ == "__main__":
    main()
