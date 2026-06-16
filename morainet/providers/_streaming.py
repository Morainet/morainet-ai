"""Pure parsers for streaming protocols (kept separate so they're unit-testable)."""

from __future__ import annotations

import json


def parse_openai_sse_line(line: str) -> str | None:
    """Extract the content delta from one OpenAI SSE line, or None to skip."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content or None


def parse_ollama_ndjson_line(line: str) -> str | None:
    """Extract the content delta from one Ollama NDJSON line, or None to skip."""
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    content = (data.get("message") or {}).get("content")
    return content or None
