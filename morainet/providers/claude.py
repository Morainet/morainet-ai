"""Anthropic Claude provider (Messages API) over httpx.

Note: implemented to the documented Messages API shape; needs live testing
against the real endpoint before production use.
"""

from __future__ import annotations

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
from morainet.providers.base import Provider

_STOP_REASON_MAP = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls"}


def to_anthropic(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Split out the system prompt and convert messages to Anthropic blocks."""
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
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
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": m.role.value, "content": m.content or ""})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


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
