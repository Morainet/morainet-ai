"""Built-in multimodal tools: vision, OCR, chart parsing, speech-to-text.

These are reference implementations. In production, swap the placeholder logic
with real vision-LLM calls (GPT-4V, Claude Vision, etc.) via the Provider.

Usage::

    from morainet.multimodal.tools import image_understand, ocr
    from morainet import tool

    agent = Agent(provider=openai, tools=[image_understand, ocr])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageAnalysisResult:
    """Structured result from an image analysis tool."""

    tool: str
    image_count: int = 0
    observations: list[str] = field(default_factory=list)
    text: str | None = None
    raw: str = ""


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def image_understand(
    image_url: str = "",
    instruction: str = "Describe this image in detail.",
    base64_data: str = "",
    media_type: str = "image/jpeg",
) -> str:
    """Analyze an image with vision-capable LLM.

    This tool describes the image to the agent so it can incorporate visual
    information into its reasoning. In production, the calling agent's
    provider handles the actual vision model call.

    Args:
        image_url: Public URL of the image to analyze.
        instruction: What to look for / what question to answer.
        base64_data: Raw base64 image data (alternative to URL).
        media_type: Media type of the base64 data (e.g. "image/png").

    Returns:
        A textual description of the image contents.

    Note:
        This is a ``@tool`` wrapper. When invoked by the agent, the agent's
        vision-capable provider processes the image. For text-only providers,
        returns a placeholder requesting multimodal capability.
    """
    source = image_url or (f"base64 image ({len(base64_data)} bytes, {media_type})")
    return (
        f"[Vision analysis requested for: {source}]\n"
        f"Instruction: {instruction}\n"
        "Note: This tool triggers vision model inference. The agent will route "
        "the actual image to the LLM for analysis."
    )


def describe_image(image_url: str = "") -> str:
    """Generate a concise natural-language caption for the image.

    Alias for ``image_understand`` with a default captioning instruction.

    Args:
        image_url: Public URL of the image.

    Returns:
        A caption string.
    """
    return image_understand(
        image_url=image_url,
        instruction="Provide a concise, one-paragraph description of this image.",
    )


def ocr(image_url: str = "", language: str = "auto") -> str:
    """Extract text from an image using optical character recognition.

    Args:
        image_url: Public URL of the image containing text.
        language: Language hint (e.g. "zh", "en", "auto").

    Returns:
        The extracted text.
    """
    source = image_url or "attached image"
    return (
        f"[OCR requested for: {source}]\n"
        f"Language: {language}\n"
        "Note: This tool triggers vision-model OCR. The agent will route "
        "the image to a multimodal LLM that supports text extraction."
    )


def chart_parse(image_url: str = "", chart_type: str = "auto") -> str:
    """Parse a chart, graph, or diagram into structured data.

    Args:
        image_url: Public URL of the chart image.
        chart_type: Type hint ("bar", "line", "pie", "scatter", "flowchart", "auto").

    Returns:
        Structured description: chart type, axes labels, data series, trends.
    """
    source = image_url or "attached chart"
    return (
        f"[Chart parsing requested for: {source}]\n"
        f"Chart type hint: {chart_type}\n"
        "Note: This tool triggers vision-model chart analysis. The agent will "
        "route the chart image to a multimodal LLM for structured extraction."
    )


def speech_to_text(
    audio_url: str = "",
    base64_data: str = "",
    format: str = "mp3",
    language: str = "auto",
) -> str:
    """Transcribe audio to text.

    Args:
        audio_url: Public URL of the audio file.
        base64_data: Raw base64 audio data.
        format: Audio format ("mp3", "wav", "ogg", "flac", "m4a").
        language: Language hint ("zh", "en", "auto").

    Returns:
        Transcription text.
    """
    source = audio_url or f"base64 audio ({len(base64_data)} bytes, {format})"
    return (
        f"[Speech-to-text requested for: {source}]\n"
        f"Language: {language}, Format: {format}\n"
        "Note: This tool triggers speech recognition. The agent will route "
        "the audio to a model that supports transcription."
    )


# ---------------------------------------------------------------------------
# Async wrapper (for real provider calls)
# ---------------------------------------------------------------------------


class MultimodalToolRunner:
    """Async runner that actually calls a vision-capable provider.

    Usage::

        runner = MultimodalToolRunner(provider)
        result = await runner.ocr("https://example.com/receipt.jpg")
    """

    def __init__(self, provider: Any) -> None:
        self.provider = provider

    async def image_understand(
        self,
        image_url: str = "",
        instruction: str = "Describe this image in detail.",
        base64_data: str = "",
        media_type: str = "image/jpeg",
    ) -> str:
        """Actually call a vision model for image analysis."""
        from morainet.core.models import Message

        if image_url:
            msg = Message.with_image_url(instruction or "Describe this image.", image_url)
        elif base64_data:
            msg = Message.with_image_base64(instruction or "Describe this image.", base64_data, media_type)
        else:
            return "Error: no image provided."

        resp = await self.provider.chat([msg])
        return resp.message.content or ""

    async def ocr(self, image_url: str = "", language: str = "auto") -> str:
        """Actually call a vision model for OCR."""
        instruction = (
            f"Extract ALL text visible in this image. "
            f"Language: {language}. Return the text only, no commentary."
        )
        return await self.image_understand(image_url=image_url, instruction=instruction)

    async def chart_parse(self, image_url: str = "", chart_type: str = "auto") -> str:
        """Actually call a vision model for chart parsing."""
        instruction = (
            f"Analyze this chart/graph. "
            f"Chart type hint: {chart_type}. "
            f"Describe: chart type, axes, data series, key trends, outliers."
        )
        return await self.image_understand(image_url=image_url, instruction=instruction)
