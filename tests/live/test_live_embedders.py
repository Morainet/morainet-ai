"""Live embedder tests — hit real embedding endpoints. Run with `pytest -m live`."""

from __future__ import annotations

import os

import httpx
import pytest

from morainet.core.models import Message
from morainet.memory import InMemoryVectorStore, LongMemory

pytestmark = pytest.mark.live


def _ollama_up() -> bool:
    try:
        base = os.getenv("MORAINET_OLLAMA_BASE_URL", "http://localhost:11434")
        return httpx.get(f"{base}/api/version", timeout=1.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="ollama not running")
async def test_ollama_embedder_live():
    from morainet.memory import OllamaEmbedder

    model = os.getenv("MORAINET_EMBED_MODEL", "nomic-embed-text")
    vec = await OllamaEmbedder(model=model).embed("hello world")
    assert isinstance(vec, list) and len(vec) > 0
    assert all(isinstance(x, float) for x in vec[:5])


@pytest.mark.skipif(not _ollama_up(), reason="ollama not running")
async def test_long_memory_with_real_embedder():
    from morainet.memory import OllamaEmbedder

    model = os.getenv("MORAINET_EMBED_MODEL", "nomic-embed-text")
    mem = LongMemory(store=InMemoryVectorStore(), embedder=OllamaEmbedder(model=model))
    await mem.add(Message.assistant(content="用户喜欢爬山和摄影"))
    hits = await mem.get_context("户外运动爱好", limit=1)
    assert hits  # semantic retrieval returns the related memory


@pytest.mark.skipif(not os.getenv("MORAINET_OPENAI_API_KEY"), reason="no OpenAI key")
async def test_openai_embedder_live():
    from morainet.memory import OpenAIEmbedder

    vec = await OpenAIEmbedder().embed("hello world")
    assert isinstance(vec, list) and len(vec) > 0
