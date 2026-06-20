"""Google Gemini provider (Generative Language API) over httpx.

Gemini matches tool results by function *name*, not by id. We therefore use
the function name as the synthetic ToolCall id so the round-trip is lossless.

Note: implemented to the documented API shape; needs live testing before
production use.
"""

from __future__ import annotations

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
from morainet.providers._streaming import parse_gemini_sse_line
from morainet.providers.base import Provider


def to_gemini(messages: list[Message]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Convert messages to Gemini GenerateContent format.

    Handles multimodal content: images as inlineData, text as text parts.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            if m.content:
                system_parts.append(
                    m.content if isinstance(m.content, str)
                    else _extract_text_gemini(m.content)
                )
        elif m.role == Role.TOOL:
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": m.tool_call_id or "",
                                "response": {"result": m.content or ""},
                            }
                        }
                    ],
                }
            )
        elif m.role == Role.ASSISTANT and m.tool_calls:
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content if isinstance(m.content, str) else _extract_text_gemini(m.content or [])})
            for tc in m.tool_calls:
                parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
            contents.append({"role": "model", "parts": parts})
        elif isinstance(m.content, list) and m.role == Role.USER:
            # Multimodal user message — build Gemini parts
            parts: list[dict[str, Any]] = []
            for item in m.content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") in ("image_url", "image"):
                    img = item.get("image_url", {})
                    url = img.get("url", "") if isinstance(img, dict) else str(img)
                    if url.startswith("data:"):
                        header, b64 = url.split(",", 1)
                        media_type = "image/jpeg"
                        if ":" in header.split(";")[0]:
                            media_type = header.split(":")[1].split(";")[0]
                        parts.append({"inlineData": {"mimeType": media_type, "data": b64}})
                    else:
                        parts.append({"fileData": {"fileUri": url}})
                elif item.get("type") == "image_base64":
                    parts.append({
                        "inlineData": {
                            "mimeType": item.get("media_type", "image/jpeg"),
                            "data": item.get("data", ""),
                        },
                    })
                elif item.get("type") == "audio":
                    audio = item.get("audio", {})
                    parts.append({
                        "inlineData": {
                            "mimeType": f"audio/{audio.get('format', 'mp3')}",
                            "data": audio.get("data", ""),
                        },
                    })
                else:
                    parts.append({"text": str(item)})
            contents.append({"role": "user", "parts": parts})
        else:
            role = "model" if m.role == Role.ASSISTANT else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m.content if isinstance(m.content, str) else _extract_text_gemini(m.content or []) if m.content else ""}],
            })

    system = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    return system, contents


def _extract_text_gemini(content: list[dict[str, Any]]) -> str:
    """Extract concatenated text from content blocks for Gemini."""
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
    candidates = data.get("candidates", [])
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    finish_reason = "stop"

    if candidates:
        cand = candidates[0]
        finish_reason = "tool_calls" if cand.get("finishReason") == "TOOL_CALL" else "stop"
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                # name doubles as the id (Gemini matches results by name)
                tool_calls.append(
                    ToolCall(id=fc["name"], name=fc["name"], arguments=fc.get("args", {}))
                )
    if tool_calls:
        finish_reason = "tool_calls"

    meta = data.get("usageMetadata", {})
    return ChatResponse(
        message=Message(
            role=Role.ASSISTANT,
            content="".join(text_parts) or None,
            tool_calls=tool_calls,
        ),
        usage=Usage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        ),
        model=model,
        finish_reason=finish_reason,
    )


class GeminiProvider(Provider):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or settings.gemini_api_key
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        if not self.api_key:
            raise AuthError("Gemini API key not set (MORAINET_GEMINI_API_KEY).")

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        system, contents = to_gemini(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = system
        if tools:
            payload["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": s["name"],
                            "description": s.get("description", ""),
                            "parameters": s["parameters"],
                        }
                        for s in tools
                    ]
                }
            ]

        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload, params={"key": self.api_key})
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code in (401, 403):
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
        system, contents = to_gemini(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = system
        if tools:
            payload["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": s["name"],
                            "description": s.get("description", ""),
                            "parameters": s["parameters"],
                        }
                        for s in tools
                    ]
                }
            ]

        url = f"{self.base_url}/v1beta/models/{self.model}:streamGenerateContent"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, params={"key": self.api_key}
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        if resp.status_code in (401, 403):
                            raise AuthError(body)
                        if resp.status_code == 429:
                            raise RateLimitError(body)
                        raise ProviderError(f"{resp.status_code}: {body}")

                    async for line in resp.aiter_lines():
                        delta = parse_gemini_sse_line(line)
                        if delta:
                            yield delta
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
