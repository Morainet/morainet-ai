"""Pure parsers for streaming protocols (kept separate so they're unit-testable)."""

from __future__ import annotations

import json
from typing import Any


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


def parse_claude_sse_event(event_type: str, data: dict[str, Any]) -> str | None:
    """Extract text delta from a Claude SSE event, or None to skip.

    Claude SSE uses named events (``event:`` line) followed by a ``data:``
    line.  This function expects the caller to accumulate the event-type
    / data pair and then pass both in.

    Supported event types that yield content:
    - ``content_block_delta`` with ``text_delta`` subtype
    """
    if event_type == "content_block_delta":
        delta = data.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text")
    return None


def parse_gemini_sse_line(line: str) -> str | None:
    """Extract content delta from one Gemini SSE line, or None to skip.

    Gemini ``streamGenerateContent`` returns SSE in the form::

        data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}

    Each ``data:`` line is self-contained.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    parts = candidates[0].get("content", {}).get("parts") or []
    if not parts:
        return None
    text = parts[0].get("text")
    return text or None
