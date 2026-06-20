"""Core data models shared across all modules.

These Pydantic models are the framework's internal lingua franca. Providers
translate them to/from vendor-specific schemas so the rest of the kernel stays
vendor-agnostic.

Message.content supports both plain ``str`` and rich ``list[dict]``
(multimodal content blocks). The framework transparently routes each format
to the appropriate provider capability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Union as UnionType

from pydantic import BaseModel, Field


Content = UnionType[str, list[dict[str, Any]], None]
"""A message's content: text string, multimodal block list, or None.

Multimodal blocks follow a vendor-neutral schema::

    [
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "https://...", "detail": "auto"}},
    ]

Providers translate this to their native multimodal format.
"""


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A turn in the conversation.

    ``content`` can be a plain string (text-only) or a list of content-part
    dicts for multimodal messages. Backward compatible — existing string-based
    code continues to work unchanged.
    """

    role: Role
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # set when role == TOOL

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str | list[dict[str, Any]]) -> "Message":
        """Create a user message.

        ``content`` may be plain text or a list of content-part dicts.
        """
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> "Message":
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)

    # ---- multimodal convenience builders ----

    @classmethod
    def multimodal_user(cls, text: str, *sources: str) -> "Message":
        """Build a multimodal user message with text + images/files.

        Args:
            text: Instruction text.
            *sources: URLs, local file paths, or data URIs. Auto-detected.

        Example::

            msg = Message.multimodal_user(
                "Describe this photo", "https://example.com/photo.jpg"
            )
        """
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for src in sources:
            if src.startswith(("http://", "https://")):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": src, "detail": "auto"},
                })
            elif src.startswith("data:image/"):
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": src, "detail": "auto"},
                })
            else:
                # Treat as local path — the caller should use content.py utilities
                parts.append({"type": "text", "text": f"[Attachment: {src}]"})
        return cls(role=Role.USER, content=parts)

    @classmethod
    def with_image_url(cls, text: str, image_url: str, detail: str = "auto") -> "Message":
        """Build a multimodal user message with one image URL.

        Example::

            msg = Message.with_image_url("What's this?", "https://example.com/img.png")
        """
        return cls(
            role=Role.USER,
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
            ],
        )

    @classmethod
    def with_image_base64(cls, text: str, b64_data: str, media_type: str = "image/jpeg") -> "Message":
        """Build a multimodal user message with a base64-encoded image.

        Example::

            import base64
            b64 = base64.b64encode(path.read_bytes()).decode()
            msg = Message.with_image_base64("Describe:", b64, "image/png")
        """
        data_uri = f"data:{media_type};base64,{b64_data}"
        return cls(
            role=Role.USER,
            content=[
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}},
            ],
        )

    # ---- content introspection helpers ----

    @property
    def text_content(self) -> str:
        """Extract a plain-text representation of the content.

        For multimodal content, concatenates text parts and generates
        placeholders for non-text parts.
        """
        c = self.content
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        parts: list[str] = []
        for item in c:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                url_info = item.get("image_url", {})
                url = url_info.get("url", "") if isinstance(url_info, dict) else str(url_info)
                parts.append(f"[Image: {url[:60]}...]" if len(url) > 60 else f"[Image: {url}]")
            elif item.get("type") == "audio":
                audio_info = item.get("audio", {})
                transcript = audio_info.get("transcript", "")
                if transcript:
                    parts.append(f"[Audio transcript: {transcript[:100]}]")
                else:
                    parts.append(f"[Audio: {audio_info.get('format', 'unknown')}]")
            elif item.get("type") == "file":
                file_info = item.get("file", {})
                parts.append(f"[File: {file_info.get('file_name', 'unknown')}]")
            else:
                parts.append(str(item))
        return " ".join(parts)

    def has_images(self) -> bool:
        """Check if this message contains image content."""
        c = self.content
        if c is None or isinstance(c, str):
            return False
        return any(
            item.get("type") in ("image_url", "image", "image_base64")
            for item in c
        )


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ChatResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    finish_reason: str = "stop"  # stop | tool_calls | length ...


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class Step(BaseModel):
    index: int
    description: str
    status: StepStatus = StepStatus.PENDING
    output: Any | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentResult(BaseModel):
    final_answer: str
    steps: list[Step] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    trace_id: str
