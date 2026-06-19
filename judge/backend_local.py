"""Local small-model backend: Qwen2.5 served by Ollama.

Talks to the Ollama server through its OpenAI-compatible endpoint
(http://localhost:11434/v1), so the same thin client works for the small judge
and the zero-shot baseline. Requires a running Ollama server with the model
pulled:  ollama pull qwen2.5:3b-instruct
"""
from __future__ import annotations

import time

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str, base_url: str = "http://localhost:11434/v1",
                 temperature: float = 0.0, max_output_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # api_key is required by the client but ignored by Ollama.
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    @retry(stop=stop_after_attempt(4),
           wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
    def chat(self, prompt: str) -> dict:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "text": text,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "latency_s": latency,
        }


def build_backend(cfg_section: dict) -> OllamaBackend:
    return OllamaBackend(
        model=cfg_section["model"],
        base_url=cfg_section.get("base_url", "http://localhost:11434/v1"),
        temperature=cfg_section.get("temperature", 0.0),
        max_output_tokens=cfg_section.get("max_output_tokens", 512),
    )
