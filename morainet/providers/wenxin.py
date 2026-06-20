"""Baidu Wenxin (ERNIE) provider — dual-mode: native OAuth + OpenAI-compatible.

Wenxin Yiyan (文心一言) supports two access modes:
1. **Native OAuth mode**: API key + secret key → access token, native endpoint.
2. **OpenAI-compatible mode**: Same API key, but through an OpenAI-compatible
   endpoint. Simpler for integrators who just need chat completions.

By default, this provider uses the **OpenAI-compatible mode** (simpler, less
code). If you need native mode (e.g. for ERNIE-specific features like plugins,
system memories), set ``native_mode=True``.

API docs: https://cloud.baidu.com/doc/WENXINWORKSHOP/index.html
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from morainet.config import settings
from morainet.core.models import ChatResponse, Message, ToolCall, Usage
from morainet.exceptions import AuthError, ProviderError, ProviderTimeoutError
from morainet.providers._streaming import parse_openai_sse_line
from morainet.providers.base import Provider


# Native-mode model → endpoint path mapping
_WENXIN_MODEL_ENDPOINTS: dict[str, str] = {
    "ernie-4.5-8k": "/chat/completions_pro",
    "ernie-4.0-8k": "/chat/completions_pro",
    "ernie-4.0-turbo-8k": "/chat/ernie-4.0-turbo-8k",
    "ernie-3.5-8k": "/chat/completions",
    "ernie-speed-8k": "/chat/ernie_speed",
    "ernie-lite-8k": "/chat/ernie-lite",
    "ernie-tiny-8k": "/chat/ernie-tiny",
}


class WenxinProvider(Provider):
    """百度文心一言 (ERNIE).

    Two modes available:
    - ``native_mode=False`` (default): OpenAI-compatible endpoint.
      Requires ``MORAINET_WENXIN_API_KEY``.
    - ``native_mode=True``: native Baidu Qianfan OAuth endpoint.
      Requires ``MORAINET_WENXIN_API_KEY`` + ``MORAINET_WENXIN_SECRET_KEY``.

    OpenAI-compatible endpoint docs:
    https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
    """

    _token_cache: tuple[str, float] | None = None  # (token, expires_at)

    def __init__(
        self,
        model: str = "ernie-4.0-turbo-8k",
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        native_mode: bool = False,
    ) -> None:
        self.model = model
        self.api_key = api_key or settings.wenxin_api_key
        self.secret_key = secret_key or settings.wenxin_secret_key
        self.base_url = (base_url or settings.wenxin_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        self.native_mode = native_mode

        if not self.api_key:
            raise AuthError(
                "Wenxin API key not set. Set MORAINET_WENXIN_API_KEY "
                "environment variable, or pass api_key= explicitly."
            )
        if native_mode and not self.secret_key:
            raise AuthError(
                "Native mode requires MORAINET_WENXIN_SECRET_KEY "
                "to obtain OAuth access tokens."
            )

    # --- native-mode OAuth token management -------------------------------

    async def _get_access_token(self) -> str:
        """Obtain or refresh the OAuth2 access token (cached in-memory)."""
        if self._token_cache is not None:
            token, expires_at = self._token_cache
            if time.time() < expires_at - 60:  # refresh 60s before expiry
                return token

        url = (
            f"https://aip.baidubce.com/oauth/2.0/token"
            f"?grant_type=client_credentials"
            f"&client_id={self.api_key}"
            f"&client_secret={self.secret_key}"
        )
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code >= 400:
            raise AuthError(f"OAuth token request failed: {resp.text}")

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise AuthError(f"No access_token in OAuth response: {data}")

        expires_in = data.get("expires_in", 2592000)  # default 30 days
        self._token_cache = (token, time.time() + expires_in)
        return token

    def _get_native_endpoint(self) -> str:
        """Resolve the native endpoint URL for the current model."""
        path = _WENXIN_MODEL_ENDPOINTS.get(self.model, f"/chat/{self.model}")
        return f"{self.base_url}{path}"

    # --- serialization ----------------------------------------------------

    def _to_openai_message(self, m: Message) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": m.role.value}
        if m.content is not None:
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
                role="assistant",  # type: ignore[arg-type]
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

    def _parse_native_response(self, data: dict[str, Any]) -> ChatResponse:
        """Parse native Wenxin response (different field names)."""
        result = data.get("result", "")
        usage_raw = data.get("usage", {})
        return ChatResponse(
            message=Message(
                role="assistant",  # type: ignore[arg-type]
                content=result or None,
            ),
            usage=Usage(
                prompt_tokens=usage_raw.get("prompt_tokens", 0),
                completion_tokens=usage_raw.get("completion_tokens", 0),
                total_tokens=usage_raw.get("total_tokens", 0),
            ),
            model=self.model,
            finish_reason="stop",
        )

    # --- API ---------------------------------------------------------------

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        if self.native_mode:
            return await self._chat_native(messages, tools)
        return await self._chat_openai_compat(messages, tools, response_format)

    async def _chat_openai_compat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
    ) -> ChatResponse:
        """OpenAI-compatible /chat/completions endpoint."""
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

        if resp.status_code in (401, 403):
            raise AuthError(resp.text)
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        return self._parse_response(resp.json())

    async def _chat_native(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> ChatResponse:
        """Native Wenxin Qianfan endpoint."""
        token = await self._get_access_token()
        user_messages = [
            {"role": m.role.value, "content": m.content or ""}
            for m in messages
            if m.role.value in ("user", "assistant")
        ]
        system_msgs = [m.content for m in messages if m.role.value == "system" and m.content]

        payload: dict[str, Any] = {"messages": user_messages}
        if system_msgs:
            payload["system"] = "\n\n".join(system_msgs)  # type: ignore[arg-type]
        if tools:
            payload["functions"] = [
                {
                    "name": s["name"],
                    "description": s.get("description", ""),
                    "parameters": s["parameters"],
                }
                for s in tools
            ]

        url = f"{self._get_native_endpoint()}?access_token={token}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code in (401, 403):
            raise AuthError(resp.text)
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")

        data = resp.json()
        if "error_code" in data:
            raise ProviderError(
                f"Wenxin error {data.get('error_code')}: {data.get('error_msg', data)}"
            )
        return self._parse_native_response(data)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        if self.native_mode:
            result = await self._chat_native(messages, tools)
            if result.message.content:
                yield result.message.content  # type: ignore[misc]
            return

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
