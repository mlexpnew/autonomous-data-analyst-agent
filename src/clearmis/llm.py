"""Optional LLM (any OpenAI-compatible endpoint). Used for two things only:
turning a question into a JSON query plan, and rephrasing verified summaries.
It never writes or runs code, and its numbers are never trusted without verification.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

PRESETS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "ollama": ("http://localhost:11434/v1", None, "qwen2.5:7b-instruct"),
}


def _post(url: str, headers: dict, body: dict) -> dict:
    import requests

    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def chat(provider: str, post: Callable = _post) -> Callable[[str, float], str | None]:
    base, key_env, model = PRESETS[provider]
    key = os.getenv(key_env) if key_env else None
    if key_env and not key:
        raise RuntimeError(f"{key_env} is not set")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def ask(prompt: str, temperature: float = 0.0) -> str | None:
        body = {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            return post(f"{base}/chat/completions", headers, body)["choices"][0]["message"][
                "content"
            ]
        except Exception:
            return None

    return ask


def extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


REWRITE = """Rewrite this monthly business summary for a busy founder in 3-5 short sentences.
Use ONLY the numbers that appear below, written exactly as they are. Do not add any new
number, percentage, date or estimate. Plain English, no headings.

{text}"""


def rewriter(ask: Callable[[str, float], str | None]) -> Callable[[str], str | None]:
    return lambda text: ask(REWRITE.format(text=text), 0.3)
