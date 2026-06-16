from __future__ import annotations

from morainet.core.models import Message, Role, ToolCall
from morainet.providers import DeepSeekProvider
from morainet.providers._streaming import parse_ollama_ndjson_line, parse_openai_sse_line
from morainet.providers.claude import parse_response as claude_parse
from morainet.providers.claude import to_anthropic
from morainet.providers.gemini import parse_response as gemini_parse
from morainet.providers.gemini import to_gemini
from morainet.providers.ollama import parse_response as ollama_parse
from morainet.providers.ollama import to_ollama


# --- Claude ----------------------------------------------------------------


def test_claude_to_anthropic_splits_system_and_tools():
    msgs = [
        Message.system("be brief"),
        Message.user("hi"),
        Message.assistant(tool_calls=[ToolCall(id="t1", name="f", arguments={"x": 1})]),
        Message.tool("result", tool_call_id="t1"),
    ]
    system, converted = to_anthropic(msgs)
    assert system == "be brief"
    assert converted[0] == {"role": "user", "content": "hi"}
    assert converted[1]["content"][0]["type"] == "tool_use"
    assert converted[2]["content"][0]["type"] == "tool_result"
    assert converted[2]["content"][0]["tool_use_id"] == "t1"


def test_claude_parse_response_tool_use():
    data = {
        "content": [
            {"type": "text", "text": "ok"},
            {"type": "tool_use", "id": "abc", "name": "f", "input": {"x": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 7, "output_tokens": 3},
        "model": "claude-x",
    }
    resp = claude_parse(data, "claude-x")
    assert resp.message.content == "ok"
    assert resp.message.tool_calls[0].name == "f"
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.total_tokens == 10


# --- Gemini ----------------------------------------------------------------


def test_gemini_to_gemini_roles_and_system():
    msgs = [Message.system("sys"), Message.user("hi"), Message.assistant(content="yo")]
    system, contents = to_gemini(msgs)
    assert system["parts"][0]["text"] == "sys"
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"


def test_gemini_parse_function_call():
    data = {
        "candidates": [
            {
                "content": {"parts": [{"functionCall": {"name": "f", "args": {"x": 1}}}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
    }
    resp = gemini_parse(data, "gemini-x")
    assert resp.message.tool_calls[0].name == "f"
    assert resp.message.tool_calls[0].id == "f"  # name doubles as id
    assert resp.finish_reason == "tool_calls"


# --- Ollama ----------------------------------------------------------------


def test_ollama_roundtrip():
    msgs = [Message.assistant(tool_calls=[ToolCall(id="x", name="f", arguments={"a": 1})])]
    converted = to_ollama(msgs)
    assert converted[0]["tool_calls"][0]["function"]["name"] == "f"


def test_ollama_parse_response():
    data = {
        "message": {"content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
        "prompt_eval_count": 4,
        "eval_count": 2,
        "model": "llama",
    }
    resp = ollama_parse(data, "llama")
    assert resp.message.tool_calls[0].name == "f"
    assert resp.message.role == Role.ASSISTANT
    assert resp.usage.total_tokens == 6


# --- DeepSeek --------------------------------------------------------------


def test_deepseek_defaults():
    p = DeepSeekProvider(api_key="x")
    assert p.model == "deepseek-chat"
    assert p.base_url.endswith("deepseek.com/v1")


# --- Streaming parsers -----------------------------------------------------


def test_openai_sse_parser():
    assert parse_openai_sse_line('data: {"choices":[{"delta":{"content":"Hi"}}]}') == "Hi"
    assert parse_openai_sse_line("data: [DONE]") is None
    assert parse_openai_sse_line("") is None
    assert parse_openai_sse_line('data: {"choices":[{"delta":{}}]}') is None


def test_ollama_ndjson_parser():
    assert parse_ollama_ndjson_line('{"message":{"content":"Hi"},"done":false}') == "Hi"
    assert parse_ollama_ndjson_line('{"message":{"content":""},"done":true}') is None
    assert parse_ollama_ndjson_line("") is None
