"""Tests for morainet.reasoning.context_compressor."""

from __future__ import annotations



from morainet.core.models import ChatResponse, Message
from morainet.reasoning.context_compressor import (
    CompressionResult,
    ContextCompressor,
    _estimate_list,
    _estimate_tokens,
    _format_history,
    _has_decision_signal,
)
from morainet.providers.mock import MockProvider


# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------

def test_compression_result():
    msg = Message.user("hello")
    result = CompressionResult(messages=[msg], stats={"strategy": "none"})
    assert result.messages == [msg]
    assert result.stats["strategy"] == "none"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_estimate_tokens():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("abc") == 1
    assert _estimate_tokens("abcdefghijklmno") > 3


def test_estimate_list():
    msgs = [Message.user("hello"), Message.assistant(content="world")]
    tokens = _estimate_list(msgs)
    assert tokens > 0


def test_format_history():
    msgs = [Message.user("hello"), Message.assistant(content="hi there")]
    result = _format_history(msgs)
    assert "[user] hello" in result
    assert "[assistant] hi there" in result


def test_format_history_with_tool_calls():
    from morainet.core.models import ToolCall
    msg = Message.assistant(tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "test"})])
    result = _format_history([msg])
    assert "search" in result
    assert "[calls:" in result


def test_format_history_with_tool_call_id():
    msg = Message.tool(content="result", tool_call_id="call_123")
    result = _format_history([msg])
    assert "call_123" in result


def test_has_decision_signal_true():
    assert _has_decision_signal("I decided to go ahead") is True
    assert _has_decision_signal("最终决定使用方案A") is True
    assert _has_decision_signal("The conclusion is clear") is True
    assert _has_decision_signal("Found the answer") is True
    assert _has_decision_signal("发现了一个问题") is True
    assert _has_decision_signal("Therefore the result is 42") is True
    assert _has_decision_signal("因此答案是42") is True
    assert _has_decision_signal("Error: something went wrong") is True
    assert _has_decision_signal("需要进一步分析") is True


def test_has_decision_signal_false():
    assert _has_decision_signal("Hello how are you") is False
    assert _has_decision_signal("OK") is False
    assert _has_decision_signal("Let me think...") is False
    assert _has_decision_signal(None) is False
    assert _has_decision_signal("") is False


# ---------------------------------------------------------------------------
# ContextCompressor: construction
# ---------------------------------------------------------------------------

def test_compressor_defaults():
    cc = ContextCompressor()
    assert cc.keep_recent == 4
    assert cc.model_max_tokens == 128000
    assert cc.safe_margin == 0.90


def test_compressor_token_budget_derived():
    cc = ContextCompressor(model_max_tokens=100000, safe_margin=0.80)
    assert cc.token_budget == 80000


def test_compressor_explicit_token_budget():
    cc = ContextCompressor(token_budget=5000)
    assert cc.token_budget == 5000


def test_compressor_set_token_budget():
    cc = ContextCompressor()
    cc.set_token_budget(3000)
    assert cc.token_budget == 3000


def test_compressor_summary_property():
    cc = ContextCompressor()
    assert cc.summary is None


def test_compressor_key_facts():
    cc = ContextCompressor()
    assert cc.key_facts == []


# ---------------------------------------------------------------------------
# ContextCompressor: compress — fast path (within budget)
# ---------------------------------------------------------------------------

async def test_compress_within_budget():
    """When messages fit within budget, return as-is."""
    cc = ContextCompressor(token_budget=100000)
    msgs = [Message.user("short message")]
    result = await cc.compress(msgs)
    assert result.messages == msgs
    assert result.stats["strategy"] == "none"


# ---------------------------------------------------------------------------
# ContextCompressor: compress — truncate (no provider)
# ---------------------------------------------------------------------------

async def test_compress_truncate_no_provider():
    """Without provider, falls through to truncation."""
    # Use very small budget to force compression past threshold
    cc = ContextCompressor(token_budget=1, keep_recent=1)
    msgs = [
        Message.system("system prompt"),
        Message.user("msg1"),
        Message.user("msg2"),
        Message.user("msg3"),
        Message.user("msg4"),
    ]
    result = await cc.compress(msgs)
    assert result.stats["strategy"] == "truncate"
    # System prompt preserved + keep_recent
    assert len(result.messages) >= 1


# ---------------------------------------------------------------------------
# ContextCompressor: extract_key_facts
# ---------------------------------------------------------------------------

def test_extract_key_facts_keeps_tool_messages():
    from morainet.core.models import ToolCall
    cc = ContextCompressor()
    msgs = [
        Message.user("hello"),
        Message.assistant(tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})]),
        Message.tool(content="found stuff", tool_call_id="c1"),
        Message.assistant(content="I decided to use the results"),
        Message.assistant(content="OK"),  # no decision signal
    ]
    facts = cc.extract_key_facts(msgs)
    # Tool message kept, assistant with tool_calls kept, user kept, decision signal assistant kept
    assert len(facts) >= 3


# ---------------------------------------------------------------------------
# ContextCompressor: build_fact_index (no provider)
# ---------------------------------------------------------------------------

async def test_build_fact_index_no_provider():
    cc = ContextCompressor(provider=None)
    result = await cc.build_fact_index([Message.user("hello")])
    assert result == []


async def test_build_fact_index_empty():
    cc = ContextCompressor()
    result = await cc.build_fact_index([])
    assert result == []


async def test_build_fact_index_with_provider():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="- fact one\n- fact two\n  \n- fact three\n-ab")
    ))
    cc = ContextCompressor(provider=provider)
    result = await cc.build_fact_index([Message.user("important info"), Message.assistant(content="response")])
    assert "fact one" in result
    assert "fact two" in result
    assert "fact three" in result
    assert "ab" not in result  # too short (<=3 chars stripped)


# ---------------------------------------------------------------------------
# ContextCompressor: _summarize_compress
# ---------------------------------------------------------------------------

async def test_summarize_compress_no_provider():
    cc = ContextCompressor(provider=None, keep_recent=2)
    result = await cc._summarize_compress(
        [Message.system("sys")],
        [Message.user("m1"), Message.user("m2"), Message.user("m3")],
        budget=10,
    )
    assert result is None


async def test_summarize_compress_no_old_messages():
    cc = ContextCompressor(provider=MockProvider(), keep_recent=5)
    result = await cc._summarize_compress(
        [],
        [Message.user("only one")],
        budget=10000,
    )
    assert result is None


async def test_summarize_compress_with_provider():
    provider = MockProvider(handler=lambda m, t: ChatResponse(
        message=Message.assistant(content="compressed summary")
    ))
    cc = ContextCompressor(provider=provider, keep_recent=1)
    msgs = [
        Message.user("long conversation message 1"),
        Message.assistant(content="long conversation response 1"),
        Message.user("recent message"),
    ]
    result = await cc._summarize_compress(
        [Message.system("sys")],
        msgs,
        budget=100000,
    )
    assert result is not None
    assert result.stats["strategy"] == "summarize"
    assert cc.summary == "compressed summary"


# ---------------------------------------------------------------------------
# ContextCompressor: _key_fact_compress
# ---------------------------------------------------------------------------

async def test_key_fact_compress_within_budget():
    cc = ContextCompressor()
    result = await cc._key_fact_compress(
        [Message.system("sys")],
        [Message.user("short")],
        budget=100000,
    )
    assert result is not None
    assert result.stats["strategy"] == "key_facts"


async def test_key_fact_compress_over_budget():
    cc = ContextCompressor()
    msgs = [Message.user(f"msg{i} with lots of padding text to increase token count") for i in range(50)]
    result = await cc._key_fact_compress(
        [Message.system("sys")],
        msgs,
        budget=1,  # impossibly small
    )
    assert result is None
