from __future__ import annotations

from morainet import Agent, tool
from morainet.core.models import ChatResponse, Message, StepStatus
from morainet.providers import MockProvider
from morainet.reasoning import ReActStrategy
from morainet.reasoning.react import parse_action, parse_final_answer


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
