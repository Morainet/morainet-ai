"""Anthropic Claude provider (Messages API) over httpx.

Note: implemented to the documented Messages API shape; needs live testing
against the real endpoint before production use.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from morainet.config import settings
from morainet.core.models import ChatResponse, Message, Role, ToolCall, Usage
from morainet.exceptions import (
    AuthError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from morainet.providers._streaming import parse_claude_sse_event
from morainet.providers.base import Provider

_STOP_REASON_MAP = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def to_anthropic(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split out the system prompt and convert messages to Anthropic content blocks.

    For multimodal user messages with images, builds Anthropic image source blocks.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            if m.content:
                system_parts.append(
                    m.content if isinstance(m.content, str)
                    else _extract_text(m.content)
                )
        elif m.role == Role.TOOL:
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content or "",
                        }
                    ],
                }
            )
        elif m.role == Role.ASSISTANT and m.tool_calls:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content if isinstance(m.content, str) else _extract_text(m.content or [])})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            converted.append({"role": "assistant", "content": blocks})
        elif isinstance(m.content, list) and m.role == Role.USER:
            # Multimodal user message — build Anthropic content blocks
            _blocks: list[dict[str, Any]] = []
            for item in m.content:
                if item.get("type") == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") in ("image_url", "image"):
                    img = item.get("image_url", {})
                    url = img.get("url", "") if isinstance(img, dict) else str(img)
                    if url.startswith("data:"):
                        header, b64 = url.split(",", 1)
                        media_type = "image/jpeg"
                        if ":" in header.split(";")[0]:
                            media_type = header.split(":")[1].split(";")[0]
                        blocks.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": b64},
                        })
                    else:
                        blocks.append({
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        })
                elif item.get("type") == "image_base64":
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.get("media_type", "image/jpeg"),
                            "data": item.get("data", ""),
                        },
                    })
                elif item.get("type") == "audio":
                    audio = item.get("audio", {})
                    transcript = audio.get("transcript", "")
                    if transcript:
                        blocks.append({"type": "text", "text": f"[Audio transcript]: {transcript}"})
                    else:
                        blocks.append({"type": "text", "text": f"[Audio: {audio.get('format', 'unknown')}]"})
                else:
                    blocks.append({"type": "text", "text": str(item)})
            converted.append({"role": "user", "content": blocks})
        else:
            converted.append({
                "role": m.role.value,
                "content": m.content if isinstance(m.content, str) else _extract_text(m.content or []) if m.content else "",
            })

    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Extract concatenated text from a content block list."""
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif item.get("type") in ("image_url", "image"):
            parts.append("[Image]")
        elif item.get("type") == "audio":
            a = item.get("audio", {})
            parts.append(a.get("transcript", "[Audio]"))
        elif item.get("type") == "file":
            f = item.get("file", {})
            parts.append(f"[File: {f.get('file_name', 'unknown')}]")
    return " ".join(parts)


def parse_response(data: dict[str, Any], model: str) -> ChatResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                ToolCall(id=block["id"], name=block["name"], arguments=block.get("input", {}))
            )

    usage_raw = data.get("usage", {})
    prompt = usage_raw.get("input_tokens", 0)
    completion = usage_raw.get("output_tokens", 0)
    return ChatResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
        model=data.get("model", model),
        finish_reason=_STOP_REASON_MAP.get(data.get("stop_reason", ""), "stop"),
    )


class ClaudeProvider(Provider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        timeout: float | None = None,
        api_version: str = "2023-06-01",
    ) -> None:
        self.model = model
        self.api_key = api_key or settings.anthropic_api_key
        self.base_url = (base_url or settings.anthropic_base_url).rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout or settings.request_timeout
        self.api_version = api_version
        if not self.api_key:
            raise AuthError("Anthropic API key not set (MORAINET_ANTHROPIC_API_KEY).")

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        system, converted = to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "input_schema": s["parameters"],
                }
                for s in tools
            ]

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/messages", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code == 401:
            raise AuthError(resp.text)
        if resp.status_code == 429:
            raise RateLimitError(resp.text)
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        return parse_response(resp.json(), self.model)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        system, converted = to_anthropic(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "input_schema": s["parameters"],
                }
                for s in tools
            ]

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/v1/messages", json=payload, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if resp.status_code == 401:
                            raise AuthError(body)
                        if resp.status_code == 429:
                            raise RateLimitError(body)
                        raise ProviderError(f"{resp.status_code}: {body}")

                    event_type: str | None = None
                    async for line in resp.aiter_lines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        if stripped.startswith("event:"):
                            event_type = stripped[len("event:"):].strip()
                        elif stripped.startswith("data:"):
                            payload_str = stripped[len("data:"):].strip()
                            try:
                                data = json.loads(payload_str)
                            except json.JSONDecodeError:
                                continue
                            delta = parse_claude_sse_event(event_type or "", data)
                            if delta:
                                yield delta
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
