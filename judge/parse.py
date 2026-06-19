"""Robust parsing of model outputs into numeric scores.

The small open model occasionally emits malformed output, so rubric parsing
tries (1) a JSON object, then (2) a per-dimension regex fallback. Both judges
share this parser so their outputs are handled identically.
"""
from __future__ import annotations

import json
import re

from common import JUDGE_DIMS


def _clamp_int(value) -> int | None:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, v))


def parse_rubric_scores(text: str) -> dict | None:
    """Extract the four 1-5 judge scores from a rubric+CoT response.

    Returns a dict {dim: int} for all four dimensions, or None if it cannot
    recover a complete, valid set of scores.
    """
    if not text:
        return None

    # (1) Try every {...} block, preferring the last (the instructed final line).
    for block in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        scores = {d: _clamp_int(obj.get(d)) for d in JUDGE_DIMS}
        if all(scores[d] is not None for d in JUDGE_DIMS):
            return scores

    # (2) Regex fallback: find "dimension: number" anywhere in the text.
    scores = {}
    for dim in JUDGE_DIMS:
        m = re.search(rf'"?{dim}"?\s*[:=]\s*"?([1-5])', text, flags=re.IGNORECASE)
        if m:
            scores[dim] = _clamp_int(m.group(1))
    if all(scores.get(d) is not None for d in JUDGE_DIMS):
        return scores

    return None


def parse_single_score(text: str) -> int | None:
    """Extract a single 1-5 score from a direct-scoring response."""
    if not text:
        return None
    m = re.search(r"[1-5]", text)
    return _clamp_int(m.group(0)) if m else None
