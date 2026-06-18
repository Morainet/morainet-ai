"""Ollama provider (native /api/chat) over httpx — for local models.

No API key required. Tool calling is supported by recent Ollama versions.
Ollama returns tool-call arguments as an object and assigns no call id, so we
synthesize ids.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from morainet.config import settings
from morainet.core.models import ChatResponse, Message, Role, ToolCall, Usage
from morainet.exceptions import ProviderError, ProviderTimeoutError
from morainet.providers._streaming import parse_ollama_ndjson_line
from morainet.providers.base import Provider


def to_ollama(messages: list[Message]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for m in messages:
        msg: dict[str, Any] = {"role": m.role.value, "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
            ]
        converted.append(msg)
    return converted


def parse_response(data: dict[str, Any], model: str) -> ChatResponse:
    raw = data.get("message", {})
    tool_calls: list[ToolCall] = []
    for i, tc in enumerate(raw.get("tool_calls") or []):
        fn = tc.get("function", {})
        tool_calls.append(
            ToolCall(id=f"call_{i}", name=fn.get("name", ""), arguments=fn.get("arguments", {}))
        )

    prompt = data.get("prompt_eval_count", 0)
    completion = data.get("eval_count", 0)
    return ChatResponse(
        message=Message(
            role=Role.ASSISTANT,
            content=raw.get("content") or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
        model=data.get("model", model),
        finish_reason="tool_calls" if tool_calls else "stop",
    )


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama(messages),
            "stream": False,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": s} for s in tools]
        if response_format:
            payload["format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        return parse_response(resp.json(), self.model)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_ollama(messages),
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise ProviderError(f"{resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        delta = parse_ollama_ndjson_line(line)
                        if delta:
                            yield delta
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
