"""Offline (hermetic) tests for HTTP-based providers.

These exercise ``OpenAIProvider`` and ``WenxinProvider`` without any network
access by monkeypatching ``httpx.AsyncClient`` with an in-memory fake. This
covers the request/response serialization, error mapping (401/403/429/400/
>=400), native OAuth token handling, and SSE streaming — all core paths of
the multi-provider support that previously had no regression protection.
"""

from __future__ import annotations

import json

import httpx
import pytest

from morainet.core.models import Message, Role, ToolCall
from morainet.exceptions import (
    AuthError,
    ContextLengthError,
    ProviderError,
    RateLimitError,
)
from morainet.providers.openai import OpenAIProvider
from morainet.providers.wenxin import WenxinProvider


# ---------------------------------------------------------------------------
# Fake httpx transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data: dict | None = None,
                 text: str = "", lines: list[str] | None = None) -> None:
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self._lines = lines or []

    def json(self) -> dict:
        return self._json


class FakeStream:
    def __init__(self, resp: FakeResponse) -> None:
        self._resp = resp
        self.status_code = resp.status_code

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aread(self) -> bytes:
        return self._resp.text.encode("utf-8")

    async def aiter_lines(self):
        for line in self._resp._lines:
            yield line


class FakeClient:
    def __init__(self, resp: FakeResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, *args: object, **kwargs: object) -> FakeResponse:
        return self._resp

    def stream(self, *args: object, **kwargs: object) -> FakeStream:
        return FakeStream(self._resp)


@pytest.fixture
def patch_httpx(monkeypatch: pytest.MonkeyPatch):
    """Replace ``httpx.AsyncClient`` so every request returns a controllable response."""
    state: dict = {"resp": FakeResponse()}

    def factory(*args: object, **kwargs: object) -> FakeClient:
        return FakeClient(state["resp"])

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


@pytest.fixture(autouse=True)
def _reset_wenxin_token_cache():
    WenxinProvider._token_cache = None
    yield
    WenxinProvider._token_cache = None


# ---------------------------------------------------------------------------
# OpenAIProvider — serialization & parsing (no network)
# ---------------------------------------------------------------------------


def test_openai_to_openai_message_text():
    p = OpenAIProvider(api_key="k")
    d = p._to_openai_message(Message(role=Role.USER, content="hello"))
    assert d == {"role": "user", "content": "hello"}


def test_openai_to_openai_message_with_tool_calls():
    p = OpenAIProvider(api_key="k")
    msg = Message(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[ToolCall(id="c1", name="f", arguments={"x": 1})],
    )
    d = p._to_openai_message(msg)
    assert d["tool_calls"][0]["id"] == "c1"
    assert d["tool_calls"][0]["function"]["name"] == "f"
    assert d["tool_calls"][0]["function"]["arguments"] == json.dumps({"x": 1})


def test_openai_parse_response_basic():
    p = OpenAIProvider(api_key="k")
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "model": "gpt-4",
    }
    r = p._parse_response(data)
    assert r.message.content == "hi"
    assert r.usage.total_tokens == 3
    assert r.model == "gpt-4"
    assert r.finish_reason == "stop"


def test_openai_parse_response_tool_calls():
    p = OpenAIProvider(api_key="k")
    data = {
        "choices": [{"message": {"role": "assistant",
                                 "tool_calls": [{"id": "c1",
                                                 "function": {"name": "f",
                                                              "arguments": '{"x": 1}'}}]}}],
    }
    r = p._parse_response(data)
    assert r.message.tool_calls[0].name == "f"
    assert r.message.tool_calls[0].arguments == {"x": 1}


# ---------------------------------------------------------------------------
# OpenAIProvider — chat error mapping (network faked)
# ---------------------------------------------------------------------------


async def test_openai_chat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {}, "model": "m"})
    p = OpenAIProvider(api_key="k")
    r = await p.chat([Message(role=Role.USER, content="hi")])
    assert r.message.content == "ok"


async def test_openai_chat_401_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(401, text="unauth")
    p = OpenAIProvider(api_key="k")
    with pytest.raises(AuthError):
        await p.chat([Message(role=Role.USER, content="x")])


async def test_openai_chat_429_raises_rate_limit(patch_httpx):
    patch_httpx["resp"] = FakeResponse(429, text="ratelimit")
    p = OpenAIProvider(api_key="k")
    with pytest.raises(RateLimitError):
        await p.chat([Message(role=Role.USER, content="x")])


async def test_openai_chat_400_context_length(patch_httpx):
    patch_httpx["resp"] = FakeResponse(400, text="context_length exceeded")
    p = OpenAIProvider(api_key="k")
    with pytest.raises(ContextLengthError):
        await p.chat([Message(role=Role.USER, content="x")])


async def test_openai_chat_500_raises_provider(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    p = OpenAIProvider(api_key="k")
    with pytest.raises(ProviderError):
        await p.chat([Message(role=Role.USER, content="x")])


async def test_openai_stream_yields_deltas(patch_httpx):
    lines = [
        'data: {"choices":[{"delta":{"content":"He"}}]}',
        'data: {"choices":[{"delta":{"content":"llo"}}]}',
        "data: [DONE]",
    ]
    patch_httpx["resp"] = FakeResponse(200, lines=lines)
    p = OpenAIProvider(api_key="k")
    out = [c async for c in p.stream([Message(role=Role.USER, content="hi")])]
    assert out == ["He", "llo"]


# ---------------------------------------------------------------------------
# WenxinProvider — serialization & parsing (no network)
# ---------------------------------------------------------------------------


def test_wenxin_to_openai_message():
    p = WenxinProvider(api_key="k")
    d = p._to_openai_message(Message(role=Role.USER, content="hi"))
    assert d["role"] == "user"


def test_wenxin_parse_response():
    p = WenxinProvider(api_key="k")
    data = {
        "choices": [{"message": {"role": "assistant", "content": "w"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "model": "m",
    }
    r = p._parse_response(data)
    assert r.message.content == "w"
    assert r.usage.total_tokens == 3


def test_wenxin_parse_native_response():
    p = WenxinProvider(api_key="k")
    r = p._parse_native_response({"result": "native out", "usage": {"total_tokens": 5}})
    assert r.message.content == "native out"
    assert r.usage.total_tokens == 5


def test_wenxin_native_endpoint_mapping():
    p = WenxinProvider(api_key="k", model="ernie-4.0-8k", base_url="https://x.com/")
    assert p._get_native_endpoint() == "https://x.com/chat/completions_pro"

    p2 = WenxinProvider(api_key="k", model="unknown-model", base_url="https://x.com")
    assert p2._get_native_endpoint() == "https://x.com/chat/unknown-model"


# ---------------------------------------------------------------------------
# WenxinProvider — chat (OpenAI-compatible) error mapping
# ---------------------------------------------------------------------------


async def test_wenxin_chat_compat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {}, "model": "m"})
    p = WenxinProvider(api_key="k")
    r = await p.chat([Message(role=Role.USER, content="hi")])
    assert r.message.content == "ok"


async def test_wenxin_chat_compat_403_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(403, text="forbidden")
    p = WenxinProvider(api_key="k")
    with pytest.raises(AuthError):
        await p.chat([Message(role=Role.USER, content="x")])


async def test_wenxin_chat_compat_500_raises_provider(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="e")
    p = WenxinProvider(api_key="k")
    with pytest.raises(ProviderError):
        await p.chat([Message(role=Role.USER, content="x")])


# ---------------------------------------------------------------------------
# WenxinProvider — native OAuth mode
# ---------------------------------------------------------------------------


async def test_wenxin_native_token_then_chat(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"access_token": "TKN", "expires_in": 3600})
    p = WenxinProvider(api_key="k", secret_key="s", native_mode=True)
    tok = await p._get_access_token()
    assert tok == "TKN"

    patch_httpx["resp"] = FakeResponse(200, {"result": "native", "usage": {"total_tokens": 1}})
    r = await p.chat([Message(role=Role.USER, content="hi")])
    assert r.message.content == "native"


async def test_wenxin_native_token_cached(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"access_token": "TKN", "expires_in": 999999})
    p = WenxinProvider(api_key="k", secret_key="s", native_mode=True)
    assert await p._get_access_token() == "TKN"
    # Second call must hit cache (no second request needed) — just ensure it returns same.
    assert await p._get_access_token() == "TKN"


async def test_wenxin_native_error_code_raises(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"access_token": "T", "expires_in": 1})
    p = WenxinProvider(api_key="k", secret_key="s", native_mode=True)
    await p._get_access_token()
    patch_httpx["resp"] = FakeResponse(200, {"error_code": 1, "error_msg": "bad"})
    with pytest.raises(ProviderError):
        await p.chat([Message(role=Role.USER, content="hi")])


async def test_wenxin_native_403_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"access_token": "T", "expires_in": 1})
    p = WenxinProvider(api_key="k", secret_key="s", native_mode=True)
    await p._get_access_token()
    patch_httpx["resp"] = FakeResponse(403, text="forbidden")
    with pytest.raises(AuthError):
        await p.chat([Message(role=Role.USER, content="hi")])
