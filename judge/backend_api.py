"""Large proprietary-judge backend: Google Gemini via the google-genai SDK.

A simple client-side rate limiter keeps us under the free-tier request/min cap,
and transient errors (including 429 rate-limit) are retried with exponential
backoff. The Gemini key is read from the env var named in config.large_judge.
"""
from __future__ import annotations

import os
import time

from google import genai
from google.genai import types
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiBackend:
    name = "gemini"

    def __init__(self, model: str, api_key_env: str = "GEMINI_API_KEY",
                 temperature: float = 0.0, max_output_tokens: int = 512,
                 rpm_limit: int = 12):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Gemini API key not found in env var '{api_key_env}'. "
                f"Set it (e.g. in .env) before running the large judge.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.min_interval = (60.0 / rpm_limit) if rpm_limit else 0.0
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.perf_counter() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    @retry(stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def chat(self, prompt: str) -> dict:
        self._throttle()
        cfg_kwargs = dict(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        # Only the 2.5 "thinking" models accept thinking_config; disable thinking
        # there so output tokens go to the visible answer (the prompt itself
        # elicits the chain-of-thought). 2.0 models reject this field.
        if "2.5" in self.model:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        t0 = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        latency = time.perf_counter() - t0
        self._last_call = time.perf_counter()
        text = resp.text or ""
        um = resp.usage_metadata
        return {
            "text": text,
            "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(um, "total_token_count", 0) or 0,
            "latency_s": latency,
        }


class GroqBackend:
    """Large open-model judge: Llama-3.3-70B served by Groq's OpenAI-compatible
    API. Free tier with generous limits; a client-side throttle stays under the
    requests/min cap and transient errors are retried with backoff."""

    name = "groq"

    def __init__(self, model: str, api_key_env: str = "GROQ_API_KEY",
                 base_url: str = "https://api.groq.com/openai/v1",
                 temperature: float = 0.0, max_output_tokens: int = 512,
                 rpm_limit: int = 25):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Groq API key not found in env var '{api_key_env}'. "
                f"Set it (e.g. in .env) before running the large judge.")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.min_interval = (60.0 / rpm_limit) if rpm_limit else 0.0
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.perf_counter() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    @retry(stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def chat(self, prompt: str) -> dict:
        self._throttle()
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        latency = time.perf_counter() - t0
        self._last_call = time.perf_counter()
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return {
            "text": text,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "latency_s": latency,
        }


def build_backend(cfg_section: dict):
    """Construct the large-judge backend selected by cfg_section['provider']."""
    provider = cfg_section.get("provider", "gemini")
    if provider == "gemini":
        return GeminiBackend(
            model=cfg_section["model"],
            api_key_env=cfg_section.get("api_key_env", "GEMINI_API_KEY"),
            temperature=cfg_section.get("temperature", 0.0),
            max_output_tokens=cfg_section.get("max_output_tokens", 512),
            rpm_limit=cfg_section.get("rpm_limit", 12),
        )
    if provider == "groq":
        return GroqBackend(
            model=cfg_section["model"],
            api_key_env=cfg_section.get("api_key_env", "GROQ_API_KEY"),
            base_url=cfg_section.get("base_url", "https://api.groq.com/openai/v1"),
            temperature=cfg_section.get("temperature", 0.0),
            max_output_tokens=cfg_section.get("max_output_tokens", 512),
            rpm_limit=cfg_section.get("rpm_limit", 25),
        )
    raise ValueError(f"unknown large-judge provider: {provider}")
