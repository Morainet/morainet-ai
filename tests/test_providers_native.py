"""Offline (hermetic) tests for the native HTTP providers: Claude, Gemini, Ollama.

Exercises request/response serialization, chat + streaming happy paths, and the
full error mapping (401/403/429/>=400/HTTPError/Timeout) by monkeypatching
``httpx.AsyncClient`` with an in-memory fake — no network access.
"""

from __future__ import annotations

import httpx
import pytest

from morainet.core.models import Message, ToolCall
from morainet.exceptions import (
    AuthError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from morainet.providers.claude import ClaudeProvider, _extract_text, to_anthropic
from morainet.providers.claude import parse_response as parse_claude
from morainet.providers.gemini import GeminiProvider, to_gemini
from morainet.providers.gemini import parse_response as parse_gemini
from morainet.providers.ollama import (
    MultiOllamaResult,
    OllamaOptions,
    OllamaProvider,
    OllamaScheduler,
    multi_ollama_query,
    to_ollama,
)
from morainet.providers.ollama import parse_response as parse_ollama


# ---------------------------------------------------------------------------
# Fake httpx transport
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = "",
        lines: list[str] | None = None,
    ) -> None:
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
    state: dict = {"resp": FakeResponse()}

    def factory(*args: object, **kwargs: object) -> FakeClient:
        return FakeClient(state["resp"])

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return state


def _raise_client(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make every ``httpx.AsyncClient`` request raise ``exc``."""

    class _Client:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, *a: object, **k: object):
            raise exc

        def stream(self, *a: object, **k: object):
            raise exc

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


# ---------------------------------------------------------------------------
# Claude — serialization
# ---------------------------------------------------------------------------


def test_claude_to_anthropic_system_and_user():
    system, conv = to_anthropic([Message.system("be nice"), Message.user("hello")])
    assert system == "be nice"
    assert conv == [{"role": "user", "content": "hello"}]


def test_claude_to_anthropic_tool_result():
    _, conv = to_anthropic([Message.tool("result", "call_1")])
    assert conv[0]["content"][0]["type"] == "tool_result"
    assert conv[0]["content"][0]["tool_use_id"] == "call_1"


def test_claude_to_anthropic_assistant_tool_calls():
    msg = Message.assistant(
        content="thinking", tool_calls=[ToolCall(id="c1", name="f", arguments={"x": 1})]
    )
    _, conv = to_anthropic([msg])
    assert conv[0]["role"] == "assistant"
    assert conv[0]["content"][0]["type"] == "text"
    assert conv[0]["content"][1]["type"] == "tool_use"
    assert conv[0]["content"][1]["input"] == {"x": 1}


def test_claude_to_anthropic_multimodal_image():
    msg = Message.with_image_base64("describe", "AAAA", media_type="image/png")
    _, conv = to_anthropic([msg])
    blocks = conv[0]["content"]
    image_block = next(b for b in blocks if b["type"] == "image")
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == "AAAA"


def test_claude_extract_text_variants():
    out = _extract_text([
        {"type": "text", "text": "a"},
        {"type": "image_url"},
        {"type": "audio", "audio": {"transcript": "hi"}},
        {"type": "file", "file": {"file_name": "x.txt"}},
    ])
    assert out == "a [Image] hi [File: x.txt]"


def test_claude_parse_response_text_and_tool_use():
    data = {
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "stop_reason": "tool_use",
        "model": "m",
    }
    r = parse_claude(data, "fallback")
    assert r.message.content == "hi"
    assert r.message.tool_calls[0].id == "c1"
    assert r.usage.total_tokens == 3
    assert r.finish_reason == "tool_calls"


def test_claude_requires_api_key():
    with pytest.raises(AuthError):
        ClaudeProvider(api_key="")


# ---------------------------------------------------------------------------
# Claude — chat / stream
# ---------------------------------------------------------------------------


def _claude(**kw) -> ClaudeProvider:
    return ClaudeProvider(model="m", api_key="k", base_url="http://x", timeout=5, **kw)


async def test_claude_chat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "content": [{"type": "text", "text": "hi"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "model": "m",
        "stop_reason": "end_turn",
    })
    r = await _claude().chat([Message.user("hi")])
    assert r.message.content == "hi"
    assert r.finish_reason == "stop"


async def test_claude_chat_with_tools(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "content": [{"type": "text", "text": "hi"}], "usage": {}, "model": "m"})
    tools = [{"name": "f", "description": "d", "parameters": {"type": "object"}}]
    r = await _claude().chat([Message.user("hi")], tools=tools)
    assert r.message.content == "hi"


async def test_claude_chat_401_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(401, text="unauth")
    with pytest.raises(AuthError):
        await _claude().chat([Message.user("x")])


async def test_claude_chat_429_raises_rate_limit(patch_httpx):
    patch_httpx["resp"] = FakeResponse(429, text="slow down")
    with pytest.raises(RateLimitError):
        await _claude().chat([Message.user("x")])


async def test_claude_chat_500_raises_provider(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(ProviderError):
        await _claude().chat([Message.user("x")])


async def test_claude_chat_http_error(monkeypatch):
    _raise_client(monkeypatch, httpx.HTTPError("conn refused"))
    with pytest.raises(ProviderError):
        await _claude().chat([Message.user("x")])


async def test_claude_chat_timeout(monkeypatch):
    _raise_client(monkeypatch, httpx.TimeoutException("timeout"))
    with pytest.raises(ProviderTimeoutError):
        await _claude().chat([Message.user("x")])


async def test_claude_stream_yields_deltas(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, lines=[
        "",
        "event: message_start",
        'data: {"type":"message_start"}',
        "event: content_block_delta",
        'data: {"delta":{"type":"text_delta","text":"Hi"}}',
        "event: content_block_delta",
        'data: {"delta":{"type":"text_delta","text":"!"}}',
    ])
    out = [c async for c in _claude().stream([Message.user("hi")])]
    assert out == ["Hi", "!"]


async def test_claude_stream_401_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(401, text="unauth")
    with pytest.raises(AuthError):
        async for _ in _claude().stream([Message.user("hi")]):
            pass


# ---------------------------------------------------------------------------
# Gemini — serialization
# ---------------------------------------------------------------------------


def test_gemini_to_gemini_system_and_user():
    system, contents = to_gemini([Message.system("sys"), Message.user("hi")])
    assert system == {"parts": [{"text": "sys"}]}
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_gemini_to_gemini_tool_response():
    _, contents = to_gemini([Message.tool("res", "fname")])
    fr = contents[0]["parts"][0]["functionResponse"]
    assert fr["name"] == "fname"
    assert fr["response"] == {"result": "res"}


def test_gemini_to_gemini_assistant_tool_calls():
    msg = Message.assistant(
        content="t", tool_calls=[ToolCall(id="f", name="f", arguments={"a": 1})]
    )
    _, contents = to_gemini([msg])
    assert contents[0]["role"] == "model"
    assert contents[0]["parts"][1]["functionCall"]["name"] == "f"


def test_gemini_to_gemini_multimodal_image():
    msg = Message.with_image_base64("desc", "BBBB", media_type="image/png")
    _, contents = to_gemini([msg])
    inline = next(p for p in contents[0]["parts"] if "inlineData" in p)
    assert inline["inlineData"]["mimeType"] == "image/png"
    assert inline["inlineData"]["data"] == "BBBB"


def test_gemini_parse_response():
    data = {
        "candidates": [{
            "content": {"parts": [{"text": "hi"}, {"functionCall": {"name": "f", "args": {"a": 1}}}]},
            "finishReason": "TOOL_CALL",
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
    }
    r = parse_gemini(data, "m")
    assert r.message.content == "hi"
    assert r.message.tool_calls[0].id == "f"
    assert r.usage.total_tokens == 3
    assert r.finish_reason == "tool_calls"


def test_gemini_parse_response_empty():
    r = parse_gemini({}, "m")
    assert r.message.content is None
    assert r.finish_reason == "stop"


def test_gemini_requires_api_key():
    with pytest.raises(AuthError):
        GeminiProvider(api_key="")


# ---------------------------------------------------------------------------
# Gemini — chat / stream
# ---------------------------------------------------------------------------


def _gemini(**kw) -> GeminiProvider:
    return GeminiProvider(model="m", api_key="k", base_url="http://x", timeout=5, **kw)


async def test_gemini_chat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"totalTokenCount": 2},
    })
    r = await _gemini().chat([Message.user("hi")])
    assert r.message.content == "hi"


async def test_gemini_chat_with_tools(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    tools = [{"name": "f", "description": "d", "parameters": {"type": "object"}}]
    r = await _gemini().chat([Message.user("hi")], tools=tools)
    assert r.message.content == "hi"


async def test_gemini_chat_401_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(401, text="unauth")
    with pytest.raises(AuthError):
        await _gemini().chat([Message.user("x")])


async def test_gemini_chat_403_raises_auth(patch_httpx):
    patch_httpx["resp"] = FakeResponse(403, text="forbidden")
    with pytest.raises(AuthError):
        await _gemini().chat([Message.user("x")])


async def test_gemini_chat_429_raises_rate_limit(patch_httpx):
    patch_httpx["resp"] = FakeResponse(429, text="slow")
    with pytest.raises(RateLimitError):
        await _gemini().chat([Message.user("x")])


async def test_gemini_chat_500_raises_provider(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(ProviderError):
        await _gemini().chat([Message.user("x")])


async def test_gemini_chat_timeout(monkeypatch):
    _raise_client(monkeypatch, httpx.TimeoutException("timeout"))
    with pytest.raises(ProviderTimeoutError):
        await _gemini().chat([Message.user("x")])


async def test_gemini_stream_yields_deltas(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, lines=[
        'data: {"candidates":[{"content":{"parts":[{"text":"He"}]}}]}',
        "",
        "data: not-json",
        'data: {"candidates":[]}',
        'data: {"candidates":[{"content":{"parts":[{"text":"llo"}]}}]}',
    ])
    out = [c async for c in _gemini().stream([Message.user("hi")])]
    assert out == ["He", "llo"]


async def test_gemini_stream_500_raises_provider(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(ProviderError):
        async for _ in _gemini().stream([Message.user("hi")]):
            pass


# ---------------------------------------------------------------------------
# Ollama — serialization & helpers
# ---------------------------------------------------------------------------


def test_ollama_options_to_dict_omits_none():
    opts = OllamaOptions(num_ctx=4096, temperature=0.7)
    assert opts.to_dict() == {"num_ctx": 4096, "temperature": 0.7}
    assert "num_gpu" not in opts.to_dict()


def test_ollama_to_ollama_text():
    assert to_ollama([Message.user("hi")]) == [{"role": "user", "content": "hi"}]


def test_ollama_to_ollama_multimodal_image():
    msg = Message.with_image_url("desc", "data:image/png;base64,AAAA")
    conv = to_ollama([msg])
    assert conv[0]["content"] == "desc"
    assert conv[0]["images"] == ["AAAA"]


def test_ollama_to_ollama_tool_calls():
    msg = Message.assistant(
        content=None, tool_calls=[ToolCall(id="c", name="f", arguments={"x": 1})]
    )
    conv = to_ollama([msg])
    assert conv[0]["tool_calls"][0]["function"]["name"] == "f"


def test_ollama_parse_response_with_tool_calls():
    data = {
        "message": {"content": "hi", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
        "prompt_eval_count": 1,
        "eval_count": 2,
        "model": "m",
    }
    r = parse_ollama(data, "fb")
    assert r.message.tool_calls[0].id == "call_0"
    assert r.usage.total_tokens == 3
    assert r.finish_reason == "tool_calls"


async def test_ollama_scheduler_acquire_release():
    sched = OllamaScheduler(max_concurrency=1)
    assert sched.max_concurrency == 1
    await sched.acquire()
    sched.release()
    with sched as ctx:
        assert ctx is sched


# ---------------------------------------------------------------------------
# Ollama — chat / stream / batch
# ---------------------------------------------------------------------------


def _ollama(**kw) -> OllamaProvider:
    return OllamaProvider(model="m", base_url="http://x", timeout=5, **kw)


async def test_ollama_chat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {
        "message": {"content": "ok"}, "model": "m", "eval_count": 1, "prompt_eval_count": 1})
    r = await _ollama().chat([Message.user("hi")])
    assert r.message.content == "ok"


async def test_ollama_chat_status_error(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(ProviderError):
        await _ollama().chat([Message.user("hi")])


async def test_ollama_chat_http_error(monkeypatch):
    _raise_client(monkeypatch, httpx.HTTPError("conn refused"))
    with pytest.raises(ProviderError):
        await _ollama().chat([Message.user("hi")])


async def test_ollama_chat_timeout(monkeypatch):
    _raise_client(monkeypatch, httpx.TimeoutException("timeout"))
    with pytest.raises(ProviderTimeoutError):
        await _ollama().chat([Message.user("hi")])


async def test_ollama_stream_yields_deltas(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, lines=[
        '{"message":{"content":"He"}}',
        "",
        "not-json",
        '{"message":{"content":"llo"},"done":true}',
    ])
    out = [c async for c in _ollama().stream([Message.user("hi")])]
    assert out == ["He", "llo"]


async def test_ollama_stream_status_error(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(ProviderError):
        async for _ in _ollama().stream([Message.user("hi")]):
            pass


async def test_ollama_batch_chat_empty():
    assert await _ollama().batch_chat([]) == []


async def test_ollama_batch_chat_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"message": {"content": "ok"}, "model": "m"})
    res = await _ollama().batch_chat([[Message.user("a")], [Message.user("b")]])
    assert len(res) == 2


async def test_ollama_batch_chat_failure_raises(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    with pytest.raises(RuntimeError):
        await _ollama().batch_chat([[Message.user("a")]])


async def test_ollama_batch_generate(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"message": {"content": "ok"}, "model": "m"})
    res = await _ollama().batch_generate(["a", "b"], system_prompt="sys")
    assert len(res) == 2


async def test_multi_ollama_query_return_all(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"message": {"content": "ok"}, "model": "m"})
    providers = {
        "m1": OllamaProvider(model="m1", base_url="http://x", timeout=5),
        "m2": OllamaProvider(model="m2", base_url="http://x", timeout=5),
    }
    results = await multi_ollama_query(providers, [Message.user("hi")], return_all=True)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(r.response is not None for r in results)


async def test_multi_ollama_query_first_success(patch_httpx):
    patch_httpx["resp"] = FakeResponse(200, {"message": {"content": "ok"}, "model": "m"})
    providers = {"m1": OllamaProvider(model="m1", base_url="http://x", timeout=5)}
    result = await multi_ollama_query(providers, [Message.user("hi")])
    assert isinstance(result, MultiOllamaResult)
    assert result.response is not None


async def test_multi_ollama_query_all_fail(patch_httpx):
    patch_httpx["resp"] = FakeResponse(500, text="boom")
    providers = {"m1": OllamaProvider(model="m1", base_url="http://x", timeout=5)}
    with pytest.raises(RuntimeError):
        await multi_ollama_query(providers, [Message.user("hi")])
