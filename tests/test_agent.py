from __future__ import annotations

import pytest

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, Role, StepStatus, ToolCall, Usage
from morainet.exceptions import MaxStepsExceededError
from morainet.memory.base import Memory
from morainet.persistence import Checkpoint, InMemoryCheckpointStore
from morainet.providers import MockProvider


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: first
        b: second
    """
    return a + b


def _tool_then_answer():
    return MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                finish_reason="tool_calls",
            ),
            ChatResponse(
                message=Message.assistant(content="结果是 5"),
                usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                finish_reason="stop",
            ),
        ]
    )


async def test_direct_answer_no_tools():
    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="你好"))]
    )
    agent = Agent(provider=provider)
    result = await agent.arun("hi")
    assert result.final_answer == "你好"
    assert result.steps == []


async def test_tool_calling_loop():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    result = await agent.arun("2+3?")
    assert result.final_answer == "结果是 5"
    assert len(result.steps) == 1
    assert result.steps[0].status == StepStatus.SUCCESS
    assert result.steps[0].output == 5
    assert result.usage.total_tokens == 20  # 12 + 8


async def test_tool_error_fed_back():
    @tool
    def boom() -> str:
        """always fails."""
        raise RuntimeError("kaboom")

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="boom", arguments={})]
                ),
                finish_reason="tool_calls",
            ),
            ChatResponse(message=Message.assistant(content="抱歉，工具失败了")),
        ]
    )
    agent = Agent(provider=provider, tools=[boom])
    result = await agent.arun("run boom")
    assert result.steps[0].status == StepStatus.FAILED
    assert "kaboom" in (result.steps[0].error or "")
    assert result.final_answer == "抱歉，工具失败了"


async def test_max_steps_exceeded():
    # Always returns a tool call -> never converges.
    looping = MockProvider(
        handler=lambda messages, tools: ChatResponse(
            message=Message.assistant(
                tool_calls=[ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})]
            ),
            finish_reason="tool_calls",
        )
    )
    agent = Agent(provider=looping, tools=[add], max_steps=3)
    with pytest.raises(MaxStepsExceededError):
        await agent.arun("loop forever")


def test_sync_run_wrapper():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    result = agent.run("2+3?")
    assert result.final_answer == "结果是 5"


async def test_astream_yields_content():
    provider = MockProvider(handler=lambda m, t: ChatResponse(message=Message.assistant(content="hello")))
    agent = Agent(provider=provider)
    chunks = [c async for c in agent.astream("hi")]
    assert "".join(chunks) == "hello"


# --- as_tool ---

async def test_as_tool_returns_tool():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    t = agent.as_tool(name="my_sub_agent", description="A sub agent")
    assert t.name == "my_sub_agent"
    assert "my_sub_agent" in t.schema.get("name", "")
    assert "query" in str(t.schema.get("parameters", {}).get("properties", {}))


async def test_as_tool_invoke():
    agent = Agent(provider=_tool_then_answer(), tools=[add])
    t = agent.as_tool(name="adder", description="Adds numbers")
    result = await t.invoke({"query": "2+3?"})
    assert result == "结果是 5"


# --- system_prompt injection (_prepare_context) ---

async def test_system_prompt_injected():
    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="ok"), usage=Usage(total_tokens=3))]
    )
    agent = Agent(provider=provider, system_prompt="You are a helpful bot.")
    result = await agent.arun("hi")
    assert result.final_answer == "ok"


# --- _remember via memory ---

async def test_memory_add_called():
    store: list[Message] = []

    class ListMemory(Memory):
        async def add(self, message: Message) -> None:
            store.append(message)

        async def get_context(self, query: str, limit: int = 10) -> list[Message]:
            return []

    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="done"), usage=Usage(total_tokens=3))]
    )
    agent = Agent(provider=provider, memory=ListMemory())
    result = await agent.arun("hello")
    assert result.final_answer == "done"
    # memory.add should be called with user query and assistant answer
    assert len(store) >= 2
    assert store[0].role == Role.USER
    assert store[0].content == "hello"  # type: ignore[arg-type]
    assert store[1].role == Role.ASSISTANT
    assert store[1].content == "done"  # type: ignore[arg-type]


async def test_memory_context_injected():
    class ContextMemory(Memory):
        async def add(self, message: Message) -> None:
            pass

        async def get_context(self, query: str, limit: int = 10) -> list[Message]:
            return [Message(role=Role.USER, content="previous context")]

    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="ok"), usage=Usage(total_tokens=3))]
    )
    agent = Agent(provider=provider, memory=ContextMemory())
    result = await agent.arun("new query")
    assert result.final_answer == "ok"


# --- aresume (checkpoint resume) ---

async def test_aresume_from_checkpoint():
    store = InMemoryCheckpointStore()
    provider = MockProvider(
        responses=[
            ChatResponse(message=Message.assistant(content="resumed answer"), usage=Usage(total_tokens=5)),
        ]
    )
    agent = Agent(
        provider=provider,
        tools=[add],
        checkpoint_store=store,
    )
    checkpoint = Checkpoint(
        trace_id="trace-123",
        query="original query",
        messages=[Message.user("original query")],
        steps=[],
        cursor=3,
        usage=Usage(total_tokens=10),
    )
    await store.save(checkpoint)
    result = await agent.aresume(checkpoint)
    assert result.final_answer == "resumed answer"
    assert result.trace_id == "trace-123"


def test_resume_sync_wrapper():
    provider = MockProvider(
        responses=[ChatResponse(message=Message.assistant(content="resumed"), usage=Usage(total_tokens=5))]
    )
    agent = Agent(provider=provider, tools=[add])
    checkpoint = Checkpoint(
        trace_id="trace-xyz",
        query="orig",
        messages=[Message.user("orig")],
        steps=[],
        cursor=0,
    )
    result = agent.resume(checkpoint)
    assert result.final_answer == "resumed"


# --- astream with tool calls (covers run_tool_calls in streaming path) ---

async def test_astream_with_tool_calls():
    call_count = {"n": 0}

    def handler(messages, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 2, "b": 3})]
                ),
                usage=Usage(total_tokens=12),
                finish_reason="tool_calls",
            )
        else:
            return ChatResponse(
                message=Message.assistant(content="结果是 5"),
                usage=Usage(total_tokens=8),
                finish_reason="stop",
            )

    agent = Agent(provider=MockProvider(handler=handler), tools=[add])
    chunks = [c async for c in agent.astream("2+3?")]
    assert "结果是 5" in "".join(chunks)


async def test_astream_max_steps_exceeded():
    """Streaming with infinite tool calls should raise MaxStepsExceededError."""
    agent = Agent(
        provider=MockProvider(
            handler=lambda m, t: ChatResponse(
                message=Message.assistant(
                    tool_calls=[ToolCall(id="c", name="add", arguments={"a": 1, "b": 1})]
                ),
                usage=Usage(total_tokens=2),
                finish_reason="tool_calls",
            )
        ),
        tools=[add],
        max_steps=2,
    )
    with pytest.raises(MaxStepsExceededError):
        chunks = []
        async for c in agent.astream("loop"):
            chunks.append(c)


# --- Message: multimodal builders and content helpers ---

def test_message_multimodal_user_http():
    msg = Message.multimodal_user("Describe", "https://example.com/img.jpg")
    assert msg.role == Role.USER
    assert isinstance(msg.content, list)
    assert msg.content[0] == {"type": "text", "text": "Describe"}
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["url"] == "https://example.com/img.jpg"


def test_message_multimodal_user_local():
    msg = Message.multimodal_user("Check", "/tmp/doc.pdf")
    assert "[Attachment: /tmp/doc.pdf]" in str(msg.content)


def test_message_multimodal_user_data_uri():
    msg = Message.multimodal_user("Look", "data:image/png;base64,abc")
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_message_with_image_url():
    msg = Message.with_image_url("What?", "https://x.com/i.png", detail="high")
    assert msg.content[0] == {"type": "text", "text": "What?"}
    assert msg.content[1]["type"] == "image_url"
    assert msg.content[1]["image_url"]["detail"] == "high"


def test_message_with_image_base64():
    msg = Message.with_image_base64("Caption:", "abc123", "image/png")
    assert msg.content[1]["image_url"]["url"] == "data:image/png;base64,abc123"


def test_message_text_content_none():
    assert Message(role=Role.USER, content=None).text_content == ""


def test_message_text_content_str():
    assert Message.user("hello").text_content == "hello"


def test_message_text_content_multimodal():
    msg = Message(role=Role.USER, content=[
        {"type": "text", "text": "Describe:"},
        {"type": "image_url", "image_url": {"url": "https://x.com/img.jpg"}},
        {"type": "audio", "audio": {"transcript": "hi there", "format": "wav"}},
        {"type": "file", "file": {"file_name": "doc.pdf"}},
        {"type": "unknown"},
    ])
    text = msg.text_content
    assert "Describe:" in text
    assert "[Image:" in text
    assert "[Audio transcript: hi there]" in text
    assert "[File: doc.pdf]" in text


def test_message_text_content_long_url_truncated():
    msg = Message(role=Role.USER, content=[
        {"type": "image_url", "image_url": {"url": "https://x.com/" + "a" * 60 + ".jpg"}},
    ])
    assert "...]" in msg.text_content


def test_message_text_content_audio_no_transcript():
    msg = Message(role=Role.USER, content=[
        {"type": "audio", "audio": {"format": "wav"}},
    ])
    assert "[Audio: wav]" in msg.text_content


def test_message_has_images_true():
    msg = Message.user([
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "x"}},
    ])
    assert msg.has_images() is True


def test_message_has_images_false():
    assert Message.user("plain").has_images() is False
    assert Message(role=Role.USER, content=None).has_images() is False


def test_message_has_images_detects_variants():
    msg = Message(role=Role.USER, content=[{"type": "image", "url": "x"}])
    assert msg.has_images() is True
    msg2 = Message(role=Role.USER, content=[{"type": "image_base64", "data": "x"}])
    assert msg2.has_images() is True
