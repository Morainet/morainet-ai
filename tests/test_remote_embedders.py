"""Offline tests for network-backed embedders (httpx mocked).

Covers :class:`OllamaEmbedder` and :class:`OpenAIEmbedder` success paths and
error handling (HTTP errors, auth errors, status errors) without real network.
"""

from __future__ import annotations

import httpx
import pytest

from morainet.exceptions import AuthError, ProviderError
from morainet.memory.remote_embedders import OllamaEmbedder, OpenAIEmbedder


class _Resp:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.text = "error"
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _mock_post(monkeypatch, status_code: int, payload: dict) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(status_code, payload))


def _mock_post_raises(monkeypatch, exc: Exception) -> None:
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(exc))


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


async def test_ollama_embed_success(monkeypatch):
    _mock_post(monkeypatch, 200, {"embedding": [0.1, 0.2, 0.3]})
    emb = OllamaEmbedder(base_url="http://localhost:11434", timeout=5)
    assert await emb.embed("hi") == [0.1, 0.2, 0.3]


async def test_ollama_http_error(monkeypatch):
    _mock_post_raises(monkeypatch, httpx.HTTPError("conn refused"))
    with pytest.raises(ProviderError):
        await OllamaEmbedder(base_url="http://localhost:11434", timeout=5).embed("hi")


async def test_ollama_status_error(monkeypatch):
    _mock_post(monkeypatch, 500, {})
    with pytest.raises(ProviderError):
        await OllamaEmbedder(base_url="http://localhost:11434", timeout=5).embed("hi")


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


async def test_openai_embed_success(monkeypatch):
    _mock_post(monkeypatch, 200, {"data": [{"embedding": [0.5, 0.6]}]})
    emb = OpenAIEmbedder(api_key="sk", base_url="http://localhost", timeout=5)
    assert await emb.embed("hi") == [0.5, 0.6]


async def test_openai_auth_error(monkeypatch):
    _mock_post(monkeypatch, 401, {})
    with pytest.raises(AuthError):
        await OpenAIEmbedder(api_key="sk", base_url="http://localhost", timeout=5).embed("hi")


async def test_openai_status_error(monkeypatch):
    _mock_post(monkeypatch, 500, {})
    with pytest.raises(ProviderError):
        await OpenAIEmbedder(api_key="sk", base_url="http://localhost", timeout=5).embed("hi")


async def test_openai_http_error(monkeypatch):
    _mock_post_raises(monkeypatch, httpx.HTTPError("conn refused"))
    with pytest.raises(ProviderError):
        await OpenAIEmbedder(api_key="sk", base_url="http://localhost", timeout=5).embed("hi")


def test_openai_requires_api_key():
    with pytest.raises(AuthError):
        OpenAIEmbedder(api_key="")
