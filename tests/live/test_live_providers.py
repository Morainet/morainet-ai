"""Live integration tests — hit real provider endpoints.

Excluded from the default run (see `addopts = -m 'not live'`). To run them,
set the relevant credentials and:

    pytest -m live

Each test self-skips when its credential / service isn't available, so you can
run only the providers you have access to.
"""

from __future__ import annotations

import os

import httpx
import pytest

from morainet import Agent, tool

pytestmark = pytest.mark.live


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。

    Args:
        city: 城市名称
    """
    return f"{city}: 晴, 26°C"


def _ollama_up() -> bool:
    try:
        base = os.getenv("MORAINET_OLLAMA_BASE_URL", "http://localhost:11434")
        return httpx.get(f"{base}/api/version", timeout=1.0).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not os.getenv("MORAINET_OPENAI_API_KEY"), reason="no OpenAI key")
async def test_openai_live():
    from morainet.providers import OpenAIProvider

    agent = Agent(provider=OpenAIProvider(model="gpt-4o-mini"), tools=[get_weather])
    result = await agent.arun("用一个词回答：你好吗？")
    assert result.final_answer


@pytest.mark.skipif(not os.getenv("MORAINET_ANTHROPIC_API_KEY"), reason="no Anthropic key")
async def test_claude_live():
    from morainet.providers import ClaudeProvider

    agent = Agent(provider=ClaudeProvider(model="claude-sonnet-4-6"), tools=[get_weather])
    result = await agent.arun("上海天气怎么样？")
    assert result.final_answer


@pytest.mark.skipif(not os.getenv("MORAINET_GEMINI_API_KEY"), reason="no Gemini key")
async def test_gemini_live():
    from morainet.providers import GeminiProvider

    agent = Agent(provider=GeminiProvider(model="gemini-1.5-flash"), tools=[get_weather])
    result = await agent.arun("上海天气怎么样？")
    assert result.final_answer


@pytest.mark.skipif(not os.getenv("MORAINET_DEEPSEEK_API_KEY"), reason="no DeepSeek key")
async def test_deepseek_live():
    from morainet.providers import DeepSeekProvider

    agent = Agent(provider=DeepSeekProvider(), tools=[get_weather])
    result = await agent.arun("上海天气怎么样？")
    assert result.final_answer


@pytest.mark.skipif(not _ollama_up(), reason="ollama not running")
async def test_ollama_live():
    model = os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b")
    from morainet.providers import OllamaProvider

    agent = Agent(provider=OllamaProvider(model=model), tools=[get_weather])
    result = await agent.arun("上海天气怎么样？")
    assert result.final_answer
    # The local model should have invoked the tool.
    assert any(s.description == "get_weather" for s in result.steps)


@pytest.mark.skipif(not _ollama_up(), reason="ollama not running")
async def test_ollama_stream_live():
    model = os.getenv("MORAINET_OLLAMA_MODEL", "qwen2.5:3b")
    from morainet.providers import OllamaProvider

    agent = Agent(provider=OllamaProvider(model=model))
    chunks = [c async for c in agent.astream("用一句话介绍你自己")]
    assert "".join(chunks).strip()
