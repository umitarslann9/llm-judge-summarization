"""Shared utilities for the CENG 467 term project.

Centralizes: .env / config loading, JSONL IO, project paths, the human/judge
dimension mapping, and small helpers used across data/, baselines/, judge/, eval/.

Every script adds the repo root to sys.path and imports from here, e.g.:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common import load_env, load_config, read_jsonl, write_jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# SummEval human-annotated dimensions (the gold signal we correlate against).
HUMAN_DIMS = ["coherence", "consistency", "fluency", "relevance"]

# Dimensions the LLM judge is asked to score (rubric_cot.txt).
JUDGE_DIMS = ["faithfulness", "coherence", "relevance", "fluency"]

# The judge's "faithfulness" corresponds to SummEval "consistency" (factual
# consistency with the source); the other three names line up directly.
JUDGE_TO_HUMAN = {
    "faithfulness": "consistency",
    "coherence": "coherence",
    "relevance": "relevance",
    "fluency": "fluency",
}


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env into os.environ (no overwrite).

    Keeps real API keys out of the committed config while letting every script
    pick them up automatically, regardless of which shell launched it.
    """
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml (falls back to config.example.yaml)."""
    if path is None:
        path = ROOT / "config.yaml"
        if not path.exists():
            path = ROOT / "config.example.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_paths(cfg: dict) -> dict[str, Path]:
    p = cfg.get("paths", {})
    paths = {
        "data_dir": ROOT / p.get("data_dir", "data"),
        "results_dir": ROOT / p.get("results_dir", "results"),
        "prompts_dir": ROOT / p.get("prompts_dir", "prompts"),
    }
    paths["subset"] = paths["data_dir"] / "subset.jsonl"
    return paths


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_prompt(prompts_dir: Path, name: str) -> str:
    return (Path(prompts_dir) / name).read_text(encoding="utf-8")


def fill_prompt(template: str, **fields) -> str:
    """Substitute {name} placeholders without str.format(), so literal braces
    in the template (e.g. the JSON example in rubric_cot.txt) are left intact."""
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def truncate_words(text: str, max_words: int) -> str:
    """Light truncation of long source articles to stay within the small
    model's context window (logged so the effect is traceable)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])
