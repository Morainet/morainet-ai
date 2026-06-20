"""OpenAI (and OpenAI-compatible) provider built on httpx.

Works with any endpoint that speaks the OpenAI Chat Completions API
(OpenAI, DeepSeek, Ollama's OpenAI-compat endpoint, etc.) by overriding
``base_url``.
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
    ContextLengthError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from morainet.providers._streaming import parse_openai_sse_line
from morainet.providers.base import Provider


class OpenAIProvider(Provider):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model or settings.default_model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        if not self.api_key:
            raise AuthError("OpenAI API key not set (MORAINET_OPENAI_API_KEY).")

    # --- serialization -----------------------------------------------------

    def _to_openai_message(self, m: Message) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role.value}
        if m.content is not None:
            # Route multimodal content through the unified adapter
            if isinstance(m.content, list):
                from morainet.multimodal.provider_adapter import default_adapter
                converted = default_adapter.to_openai([m])
                msg["content"] = converted[0]["content"] if converted else ""
            else:
                msg["content"] = m.content
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        return msg

    def _parse_response(self, data: dict[str, Any]) -> ChatResponse:
        choice = data["choices"][0]
        raw = choice["message"]

        tool_calls: list[ToolCall] = []
        for tc in raw.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

        usage_raw = data.get("usage") or {}
        return ChatResponse(
            message=Message(
                role=Role.ASSISTANT,
                content=raw.get("content"),
                tool_calls=tool_calls,
            ),
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", "stop"),
        )

    # --- API ---------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": s} for s in tools]
        if response_format:
            payload["response_format"] = response_format

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code == 401:
            raise AuthError(resp.text)
        if resp.status_code == 429:
            raise RateLimitError(resp.text)
        if resp.status_code == 400 and "context_length" in resp.text:
            raise ContextLengthError(resp.text)
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        return self._parse_response(resp.json())

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_openai_message(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": s} for s in tools]
        if response_format:
            payload["response_format"] = response_format

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=headers
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise ProviderError(f"{resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        delta = parse_openai_sse_line(line)
                        if delta:
                            yield delta
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
