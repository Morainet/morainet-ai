"""Tests for morainet.providers.model_router — routing and multi-model query."""
from __future__ import annotations

import pytest

from morainet.core.models import ChatResponse, Message, Usage
from morainet.providers.mock import MockProvider
from morainet.providers.model_router import (
    ModelRouter,
    estimate_complexity,
    multi_model_query,
)


def _resp(content: str) -> ChatResponse:
    return ChatResponse(message=Message.assistant(content=content), usage=Usage(), model="mock")


def _fail_handler(messages: list[Message], tools: object) -> ChatResponse:
    raise RuntimeError("provider boom")


# --- estimate_complexity ------------------------------------------------------
def test_estimate_complexity_no_user_returns_moderate():
    assert estimate_complexity([Message.system("sys only")]) == 0.3


def test_estimate_complexity_simple_greeting():
    assert estimate_complexity([Message.user("hi")]) < 0.3


def test_estimate_complexity_code_keyword():
    score = estimate_complexity([Message.user("debug this code: def foo(): return 1")])
    assert score > 0.3


def test_estimate_complexity_within_bounds():
    score = estimate_complexity([Message.user("x" * 3000)])
    assert 0.0 <= score <= 1.0


# --- ModelRouter construction -------------------------------------------------
def test_router_requires_tiers():
    with pytest.raises(ValueError, match="At least one tier"):
        ModelRouter(tiers={})


def test_router_round_robin_pick():
    p1 = MockProvider(responses=[_resp("a")])
    p2 = MockProvider(responses=[_resp("b")])
    router = ModelRouter(tiers={"small": [p1, p2]})
    assert router._pick_provider("small") is p1
    assert router._pick_provider("small") is p2
    assert router._pick_provider("small") is p1


def test_router_select_tier_two_tiers():
    router = ModelRouter(tiers={"small": [MockProvider()], "large": [MockProvider()]})
    assert router._select_tier(0.1) == "small"
    assert router._select_tier(0.5) == "large"


def test_router_select_tier_single():
    router = ModelRouter(tiers={"only": [MockProvider()]})
    assert router._select_tier(0.9) == "only"


# --- ModelRouter.chat routing -------------------------------------------------
async def test_router_chat_routes_to_tier():
    small = MockProvider(responses=[_resp("from small")])
    router = ModelRouter(tiers={"small": [small], "large": [MockProvider()]})
    resp = await router.chat([Message.user("hello")])
    assert "small" in resp.message.content


async def test_router_fallback_on_failure():
    small = MockProvider(handler=_fail_handler)
    large = MockProvider(responses=[_resp("from large")])
    router = ModelRouter(tiers={"small": [small], "large": [large]})
    resp = await router.chat([Message.user("hello")])
    assert "large" in resp.message.content


async def test_router_all_tiers_fail_raises():
    small = MockProvider(handler=_fail_handler)
    large = MockProvider(handler=_fail_handler)
    router = ModelRouter(tiers={"small": [small], "large": [large]})
    with pytest.raises(RuntimeError, match="all tiers exhausted"):
        await router.chat([Message.user("hello")])


# --- stats --------------------------------------------------------------------
async def test_router_stats_and_summary():
    p = MockProvider(responses=[ChatResponse(
        message=Message.assistant("hi"),
        usage=Usage(prompt_tokens=1000, completion_tokens=0, total_tokens=1000),
        model="mock",
    )])
    router = ModelRouter(tiers={"small": [p]}, cost_weights={"small": 0.01})
    await router.chat([Message.user("hello")])
    assert router.stats.total_calls == 1
    assert router.stats.estimated_cost_usd > 0
    summary = router.get_stats_summary()
    assert summary["total_calls"] == 1
    router.reset_stats()
    assert router.stats.total_calls == 0


# --- stream -------------------------------------------------------------------
async def test_router_stream():
    p = MockProvider(responses=[_resp("streamed answer")])
    router = ModelRouter(tiers={"small": [p]})
    chunks = [c async for c in router.stream([Message.user("hi")])]
    assert chunks
    assert "streamed answer" in "".join(chunks)


# --- multi_model_query --------------------------------------------------------
async def test_multi_model_fastest():
    providers = [
        MockProvider(responses=[_resp("first")]),
        MockProvider(responses=[_resp("second")]),
    ]
    resp = await multi_model_query(providers, [Message.user("q")], pick_strategy="fastest")
    assert resp.message.content


async def test_multi_model_longest():
    providers = [
        MockProvider(responses=[_resp("short")]),
        MockProvider(responses=[_resp("much longer answer here")]),
    ]
    resp = await multi_model_query(providers, [Message.user("q")], pick_strategy="longest")
    assert "much longer" in resp.message.content


async def test_multi_model_all():
    providers = [
        MockProvider(responses=[_resp("first response")]),
        MockProvider(responses=[_resp("second response")]),
    ]
    resp = await multi_model_query(providers, [Message.user("q")], pick_strategy="all")
    assert "first response" in resp.message.content
    assert "second response" in resp.message.content
