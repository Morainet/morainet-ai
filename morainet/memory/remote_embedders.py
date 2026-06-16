"""Real embedding backends (network-backed).

These call external services, so `embed` does a blocking HTTP request. For the
offline default see :class:`morainet.memory.embeddings.HashEmbedder`.
"""

from __future__ import annotations

import httpx

from morainet.config import settings
from morainet.exceptions import AuthError, ProviderError
from morainet.memory.base import Embedder


class OllamaEmbedder(Embedder):
    """Local embeddings via Ollama (e.g. ``nomic-embed-text``). No API key."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout

    def embed(self, text: str) -> list[float]:
        try:
            resp = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")
        embedding: list[float] = resp.json()["embedding"]
        return embedding


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI (or OpenAI-compatible) embeddings endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or settings.openai_api_key
        self.base_url = (base_url or settings.openai_base_url).rstrip("/")
        self.timeout = timeout or settings.request_timeout
        if not self.api_key:
            raise AuthError("OpenAI API key not set (MORAINET_OPENAI_API_KEY).")

    def embed(self, text: str) -> list[float]:
        try:
            resp = httpx.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc
        if resp.status_code == 401:
            raise AuthError(resp.text)
        if resp.status_code >= 400:
            raise ProviderError(f"{resp.status_code}: {resp.text}")
        embedding: list[float] = resp.json()["data"][0]["embedding"]
        return embedding
