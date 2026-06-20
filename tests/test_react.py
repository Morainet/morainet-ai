from __future__ import annotations

import pytest

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, StepStatus
from morainet.exceptions import MaxStepsExceededError
from morainet.providers import MockProvider
from morainet.reasoning import ReActStrategy
from morainet.reasoning.react import _render_tools, parse_action, parse_final_answer


def test_parse_final_answer():
    assert parse_final_answer("Thought: x\nFinal Answer: 42") == "42"
    assert parse_final_answer("Thought: still thinking") is None


def test_parse_action():
    text = 'Thought: add them\nAction: add\nAction Input: {"a": 2, "b": 3}\n'
    parsed = parse_action(text)
    assert parsed == ("add", {"a": 2, "b": 3})


def test_parse_action_none_when_absent():
    assert parse_action("Thought: hmm\nFinal Answer: done") is None


def test_parse_action_bad_json_yields_empty_args():
    name, args = parse_action("Action: foo\nAction Input: not-json")  # type: ignore[misc]
    assert name == "foo"
    assert args == {}


def test_parse_action_json_array_yields_empty_args():
    """Non-dict JSON (e.g. a list) should yield empty args."""
    name, args = parse_action('Action: foo\nAction Input: [1, 2, 3]\n')  # type: ignore[misc]
    assert name == "foo"
    assert args == {}


def test_parse_action_json_string_yields_empty_args():
    """JSON string value (not dict) should yield empty args."""
    name, args = parse_action('Action: foo\nAction Input: "hello"\n')  # type: ignore[misc]
    assert name == "foo"
    assert args == {}


def test_render_tools_with_tools():
    @tool
    def search(query: str) -> str:
        """Search the web."""
        return f"results for {query}"

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(provider=MockProvider(), tools=[search, add])
    rendered = _render_tools(agent)
    assert "search(query)" in rendered
    assert "add(a, b)" in rendered
    assert "Search the web" in rendered
    assert "Add two numbers" in rendered


def test_render_tools_empty():
    agent = Agent(provider=MockProvider())
    rendered = _render_tools(agent)
    assert rendered == "(no tools available)"


async def test_react_system_prompt_override():
    """User can override the react system prompt via prompts dict."""
    @tool
    def echo(text: str) -> str:
        """Echo back."""
        return text

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    content='Thought: use echo\nAction: echo\nAction Input: {"text": "hello"}\n'
                )
            ),
            ChatResponse(
                message=Message.assistant(content="Thought: done\nFinal Answer: hello")
            ),
        ]
    )
    from morainet.prompts import PromptTemplate

    # Override react_system prompt
    agent = Agent(
        provider=provider,
        tools=[echo],
        strategy=ReActStrategy(),
        prompts={
            "react_system": PromptTemplate(
                name="react_system",
                template="Custom prompt with {tools}",
            ),
        },
    )
    result = await agent.arun("echo hello")
    assert result.final_answer == "hello"
    # _system_prompt returns a string; verify it uses our custom template
    system_prompt = agent.strategy._system_prompt(agent)
    assert "Custom prompt" in system_prompt


async def test_react_no_action_takes_text_as_answer():
    """When model answers without ReAct scaffold, use text as final answer."""
    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(content="The answer is 42.")
            ),
        ]
    )
    agent = Agent(provider=provider, strategy=ReActStrategy())
    result = await agent.arun("what is the answer?")
    assert "42" in result.final_answer


async def test_react_max_steps_exceeded():
    """ReAct loop that never gives Final Answer should raise MaxStepsExceededError."""
    @tool
    def echo(text: str) -> str:
        """Echo back."""
        return text

    provider = MockProvider(
        handler=lambda m, t: ChatResponse(
            message=Message.assistant(
                content='Thought: hmm\nAction: echo\nAction Input: {"text": "x"}\n'
            )
        )
    )
    agent = Agent(provider=provider, tools=[echo], strategy=ReActStrategy(max_steps=2))
    with pytest.raises(MaxStepsExceededError):
        await agent.arun("test")


async def test_react_end_to_end():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: first
            b: second
        """
        return a + b

    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    content='Thought: I should add\nAction: add\nAction Input: {"a": 2, "b": 3}\n'
                )
            ),
            ChatResponse(message=Message.assistant(content="Thought: done\nFinal Answer: 5")),
        ]
    )

    agent = Agent(provider=provider, tools=[add], strategy=ReActStrategy())
    result = await agent.arun("2+3?")
    assert result.final_answer == "5"
    assert len(result.steps) == 1
    assert result.steps[0].status == StepStatus.SUCCESS
    assert result.steps[0].output == 5
