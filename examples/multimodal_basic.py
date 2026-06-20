"""Multimodal basics: image understanding, OCR, chart parsing, audio transcription.

Demonstrates Morainet's native multimodal message abstraction and tool calling:
- Unified ContentPart types (text, image URL, image base64, audio, file)
- Message.multimodal_user() / with_image_url() convenience builders
- Built-in multimodal tools: image_understand, ocr, chart_parse, speech_to_text
- Provider-agnostic routing: same code works with GPT-4V, Claude Vision, Gemini, Ollama

Offline-safe: uses MockProvider for deterministic runs. Set
``MORAINET_OPENAI_API_KEY`` to use GPT-4V / Claude Vision for real vision.
"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
from pathlib import Path

from morainet import (
    Agent,
    MockProvider,
    Tool,
    tool,
)

# --- Simulated image data (1x1 pixel red PNG) ----------------------------


def _demo_image_b64() -> str:
    """Generate a tiny red PNG as base64 for demo purposes."""
    # Minimal valid PNG: 1x1 red pixel
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(png_bytes).decode("ascii")


# --- Example 1: Multimodal message construction --------------------------


def demo_multimodal_messages():
    """Show how to build multimodal messages with the new API."""
    print("=" * 60)
    print("1. Multimodal Message Construction")
    print("=" * 60)

    from morainet.core.models import Message

    # Method A: directory via content list of dicts
    msg_a = Message(
        role="user",
        content=[
            {"type": "text", "text": "Describe this image:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/photo.jpg", "detail": "auto"},
            },
        ],
    )
    print(f"\n  Method A (raw dicts):")
    print(f"    role={msg_a.role}, text_content={msg_a.text_content!r}")
    print(f"    has_images={msg_a.has_images()}")

    # Method B: convenience builder
    msg_b = Message.multimodal_user(
        "What's in this chart?",
        "https://example.com/chart.png",
    )
    print(f"\n  Method B (convenience):")
    print(f"    role={msg_b.role}, text_content={msg_b.text_content!r}")
    print(f"    has_images={msg_b.has_images()}")

    # Method C: with_image_url
    msg_c = Message.with_image_url("Analyze this diagram.", "https://example.com/diagram.png")
    print(f"\n  Method C (with_image_url):")
    print(f"    role={msg_c.role}, text_content={msg_c.text_content!r}")

    # Method D: with_image_base64
    b64_data = _demo_image_b64()
    msg_d = Message.with_image_base64("Describe this image.", b64_data, "image/png")
    print(f"\n  Method D (with_image_base64):")
    print(f"    role={msg_d.role}, has_images={msg_d.has_images()}")

    # Backward compatibility: plain string content still works
    msg_e = Message.user("Hello, how are you?")
    print(f"\n  Plain text (backward compat):")
    print(f"    role={msg_e.role}, content={msg_e.content!r}")
    print(f"    has_images={msg_e.has_images()}")

    # Mixed content: text + multiple images
    msg_f = Message(
        role="user",
        content=[
            {"type": "text", "text": "Compare these two screenshots:"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/before.png"},
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/after.png"},
            },
        ],
    )
    print(f"\n  Multiple images:")
    print(f"    text_content={msg_f.text_content!r}")
    print(f"    has_images={msg_f.has_images()}")


# --- Example 2: Multimodal tool registration -----------------------------


def demo_multimodal_tools():
    """Register and test built-in multimodal tools."""
    print("\n" + "=" * 60)
    print("2. Multimodal Tool Registration")
    print("=" * 60)

    from morainet.multimodal.tools import (
        chart_parse,
        describe_image,
        image_understand,
        ocr,
        speech_to_text,
    )

    tools: list[Tool] = [
        tool(image_understand),
        tool(describe_image),
        tool(ocr),
        tool(chart_parse),
        tool(speech_to_text),
    ]

    for t in tools:
        print(f"\n  Tool: {t.name}")
        print(f"    schema keys: {list(t.schema.keys())}")
        print(f"    dangerous: {t.dangerous}")

    # Invoke a tool directly to see its placeholder output
    print("\n  Running ocr(image_url='https://example.com/receipt.jpg'):")
    result = ocr(image_url="https://example.com/receipt.jpg", language="zh")
    print(f"    -> {result[:120]}...")


# --- Example 3: Agent with multimodal tools (offline mock) ----------------


async def demo_agent_with_multimodal_tools():
    """Run an agent with multimodal tools using MockProvider (offline-safe)."""
    print("\n" + "=" * 60)
    print("3. Agent with Multimodal Tools (MockProvider)")
    print("=" * 60)

    from morainet.multimodal.tools import (
        chart_parse,
        describe_image,
        image_understand,
        ocr,
        speech_to_text,
    )

    agent = Agent(
        provider=MockProvider(),
        tools=[
            image_understand,
            describe_image,
            ocr,
            chart_parse,
            speech_to_text,
        ],
        system_prompt=(
            "You are a multimodal assistant. Use image_understand, ocr, "
            "chart_parse, and speech_to_text tools to analyze media."
        ),
    )

    query = "Please run OCR on https://example.com/invoice.png and tell me what text it contains."
    print(f"\n  Query: {query}")
    result = await agent.arun(query)
    print(f"  Final answer: {result.final_answer[:200]}...")


# --- Example 4: Provider-agnostic adapter demo ----------------------------


def demo_provider_adapters():
    """Show how multimodal messages are routed to different providers."""
    print("\n" + "=" * 60)
    print("4. Provider-Agnostic Adapter Routing")
    print("=" * 60)

    from morainet.core.models import Message
    from morainet.multimodal.provider_adapter import default_adapter

    msgs = [
        Message.system("You are a visual assistant."),
        Message.with_image_url(
            "Describe this diagram.",
            "https://example.com/arch.png",
        ),
    ]

    # OpenAI format
    openai_msgs = default_adapter.to_openai(msgs)
    last_openai = openai_msgs[-1]
    print(f"\n  OpenAI format:")
    print(f"    role={last_openai['role']}")
    if isinstance(last_openai.get("content"), list):
        print(f"    content blocks: {[b['type'] for b in last_openai['content']]}")
    else:
        print(f"    content type: {type(last_openai['content']).__name__}")

    # Anthropic format
    system, anthropic_msgs = default_adapter.to_anthropic(msgs)
    print(f"\n  Anthropic format:")
    print(f"    system={system[:60] if system else None}")
    last_anthropic = anthropic_msgs[-1]
    if isinstance(last_anthropic.get("content"), list):
        print(f"    content blocks: {[b['type'] for b in last_anthropic['content']]}")

    # Gemini format
    gemini_system, gemini_contents = default_adapter.to_gemini(msgs)
    print(f"\n  Gemini format:")
    last_gemini = gemini_contents[-1]
    parts = last_gemini.get("parts", [])
    print(f"    parts: {[list(p.keys()) for p in parts]}")

    # Ollama format
    ollama_msgs = default_adapter.to_ollama(msgs)
    last_ollama = ollama_msgs[-1]
    print(f"\n  Ollama format:")
    print(f"    role={last_ollama['role']}")
    print(f"    content={last_ollama.get('content', '')[:80]}...")
    print(f"    images field: {'images' in last_ollama}")


# --- Example 5: Content API utilities ------------------------------------


def demo_content_api():
    """Demonstrate the content utility functions."""
    print("\n" + "=" * 60)
    print("5. Content API Utilities")
    print("=" * 60)

    from morainet.multimodal.content import (
        AudioPart,
        FilePart,
        ImageUrlPart,
        TextPart,
        content_has_images,
        content_has_text,
        content_to_str,
        split_text_and_images,
    )

    # Build a mixed content list
    content = [
        TextPart(text="Here is the architecture diagram:").to_dict(),
        ImageUrlPart(url="https://example.com/arch.png").to_dict(),
        TextPart(text="And here is a data flow chart:").to_dict(),
        ImageUrlPart(url="https://example.com/flow.png").to_dict(),
    ]

    print(f"\n  Plain text extraction:")
    text = content_to_str(content)
    print(f"    {text[:100]}...")

    print(f"\n  Content checks:")
    print(f"    has_text={content_has_text(content)}")
    print(f"    has_images={content_has_images(content)}")

    print(f"\n  Split into text + images:")
    texts, images = split_text_and_images(content)
    print(f"    text parts: {len(texts)}")
    print(f"    image parts: {len(images)}")
    for t in texts:
        print(f"      - {t.text}")

    # Audio and File parts
    print(f"\n  Audio part:")
    audio = AudioPart(data="base64...", format="mp3", transcript="Hello world")
    print(f"    str: {audio}")

    print(f"\n  File part:")
    f = FilePart(file_name="report.pdf", data="base64...", mime_type="application/pdf")
    print(f"    str: {f}")


# --- Example 6: Multimodal agent with real provider (production pattern) --


async def demo_production_pattern():
    """Show the production pattern: agent + vision provider + multimodal tools.

    This uses MockProvider for offline safety. For production, swap to
    OpenAIProvider(model="gpt-4o") or ClaudeProvider().
    """
    print("\n" + "=" * 60)
    print("6. Production Pattern (MockProvider)")
    print("=" * 60)

    from morainet.multimodal.tools import image_understand

    # Production-ready agent setup
    agent = Agent(
        provider=MockProvider(),
        tools=[image_understand],
        system_prompt=(
            "You are a visual QA assistant. Use image_understand to analyze "
            "images when needed. Provide detailed, accurate answers."
        ),
    )

    # Simulate a multimodal query
    # In production, the user's message would carry real image data
    query = (
        "Here is a system architecture diagram: "
        "https://example.com/architecture.png — "
        "Please analyze the diagram and explain the main components."
    )

    print(f"\n  Query: {query[:100]}...")
    result = await agent.arun(query)
    print(f"  Answer: {result.final_answer[:200]}...")


# --- Main ------------------------------------------------------------------


def main():
    demo_multimodal_messages()
    demo_multimodal_tools()
    demo_provider_adapters()
    demo_content_api()

    # Async demos
    asyncio.run(demo_agent_with_multimodal_tools())
    asyncio.run(demo_production_pattern())

    print("\n" + "=" * 60)
    print("All multimodal demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
