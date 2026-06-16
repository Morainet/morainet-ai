"""v0.3 features: ReAct strategy + multi-provider. ReAct demo runs offline.

Run:
    python examples/react_and_providers.py
"""

from __future__ import annotations

from morainet import Agent, ReActStrategy, tool
from morainet.core.models import ChatResponse, Message
from morainet.providers import MockProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first number
        b: second number
    """
    return a + b


def react_demo() -> None:
    print("=== ReAct strategy (offline) ===")
    provider = MockProvider(
        responses=[
            ChatResponse(
                message=Message.assistant(
                    content='Thought: add the numbers\nAction: add\nAction Input: {"a": 2, "b": 40}\n'
                )
            ),
            ChatResponse(message=Message.assistant(content="Thought: done\nFinal Answer: 42")),
        ]
    )
    agent = Agent(provider=provider, tools=[add], strategy=ReActStrategy())
    result = agent.run("2 加 40 等于几？")
    print("Final answer:", result.final_answer)
    print("Tool steps:", [(s.description, s.output) for s in result.steps])


def provider_swap_note() -> None:
    print("\n=== Swap providers (need credentials) ===")
    print("from morainet.providers import (")
    print("    OpenAIProvider, ClaudeProvider, GeminiProvider,")
    print("    OllamaProvider, DeepSeekProvider,")
    print(")")
    print("agent = Agent(provider=DeepSeekProvider(), tools=[add])  # OpenAI-compatible")
    print("agent = Agent(provider=OllamaProvider(model='llama3.1'))  # local, no key")


if __name__ == "__main__":
    react_demo()
    provider_swap_note()
