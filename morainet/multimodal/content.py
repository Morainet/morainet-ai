"""Rich multimodal content parts — the unified message envelope.

Models are vendor-agnostic. Provider adapters translate to/from vendor-specific
schemas (OpenAI blocks, Anthropic content, Gemini parts, Ollama images).

Usage::

    from morainet.multimodal import TextPart, ImageUrlPart, source_to_part

    # Build a multimodal user message
    parts = [
        TextPart(text="What's in this image?"),
        ImageUrlPart(url="https://example.com/photo.jpg"),
    ]
    msg = Message.user(content=[p.to_dict() for p in parts])

    # Or use the convenience builder
    msg = Message.multimodal_user("Describe this:", "https://example.com/photo.jpg")
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union as UnionType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    AUDIO = "audio"
    FILE = "file"


# ---------------------------------------------------------------------------
# Content part data classes
# ---------------------------------------------------------------------------


@dataclass
class ContentPart:
    """Base class for content parts."""

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentPart":
        raise NotImplementedError

    @property
    def type(self) -> ContentType:
        """Each subclass overrides this."""
        raise NotImplementedError


@dataclass
class TextPart(ContentPart):
    """Plain text content block."""

    text: str

    @property
    def type(self) -> ContentType:
        return ContentType.TEXT

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextPart":
        return cls(text=data.get("text", ""))

    def __str__(self) -> str:
        return self.text


@dataclass
class ImageUrlPart(ContentPart):
    """Image referenced by URL.

    Args:
        url: Public or data-URI image URL.
        detail: Resolution hint (``"auto"``, ``"low"``, ``"high"``).
    """

    url: str
    detail: str = "auto"

    @property
    def type(self) -> ContentType:
        return ContentType.IMAGE_URL

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.url, "detail": self.detail},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageUrlPart":
        image_url = data.get("image_url", {})
        if isinstance(image_url, str):
            return cls(url=image_url)
        return cls(url=image_url.get("url", ""), detail=image_url.get("detail", "auto"))

    def __str__(self) -> str:
        return f"[Image: {self.url[:80]}...]"


@dataclass
class ImageBase64Part(ContentPart):
    """Image as raw base64 bytes.

    Args:
        data: Base64-encoded image bytes (without the ``data:...`` prefix).
        media_type: e.g. ``"image/png"``, ``"image/jpeg"``.
        detail: Resolution hint.
    """

    data: str
    media_type: str = "image/jpeg"
    detail: str = "auto"

    @property
    def type(self) -> ContentType:
        return ContentType.IMAGE_BASE64

    def to_dict(self) -> dict[str, Any]:
        data_uri = f"data:{self.media_type};base64,{self.data}"
        return {
            "type": "image_url",
            "image_url": {"url": data_uri, "detail": self.detail},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageBase64Part":
        image_url = data.get("image_url", {})
        if isinstance(image_url, str):
            url = image_url
        else:
            url = image_url.get("url", "")
        # Parse data URI
        if url.startswith("data:"):
            header, encoded = url.split(",", 1)
            media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
            return cls(data=encoded, media_type=media_type)
        return cls(data=url)

    def to_data_uri(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"

    def __str__(self) -> str:
        return f"[Image base64: {len(self.data)} bytes, {self.media_type}]"


@dataclass
class AudioPart(ContentPart):
    """Audio content block.

    Args:
        data: Base64-encoded audio.
        format: Audio format (e.g. ``"mp3"``, ``"wav"``, ``"ogg"``).
        transcript: Optional transcript text for non-multimodal providers.
    """

    data: str
    format: str = "mp3"
    transcript: str | None = None

    @property
    def type(self) -> ContentType:
        return ContentType.AUDIO

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "audio",
            "audio": {
                "data": self.data,
                "format": self.format,
                **({"transcript": self.transcript} if self.transcript else {}),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioPart":
        audio = data.get("audio", {})
        return cls(
            data=audio.get("data", ""),
            format=audio.get("format", "mp3"),
            transcript=audio.get("transcript"),
        )

    def __str__(self) -> str:
        if self.transcript:
            return f"[Audio: {self.transcript[:100]}...]"
        return f"[Audio: {len(self.data)} bytes, {self.format}]"


@dataclass
class FilePart(ContentPart):
    """File attachment content block.

    Args:
        file_name: Original file name.
        data: Base64-encoded file content.
        mime_type: MIME type of the file.
    """

    file_name: str
    data: str
    mime_type: str = "application/octet-stream"

    @property
    def type(self) -> ContentType:
        return ContentType.FILE

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "file",
            "file": {
                "file_name": self.file_name,
                "data": self.data,
                "mime_type": self.mime_type,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilePart":
        f = data.get("file", {})
        return cls(
            file_name=f.get("file_name", ""),
            data=f.get("data", ""),
            mime_type=f.get("mime_type", "application/octet-stream"),
        )

    def __str__(self) -> str:
        return f"[File: {self.file_name} ({self.mime_type})]"


# ---------------------------------------------------------------------------
# Union type and content utilities
# ---------------------------------------------------------------------------

Content = UnionType[str, list[dict[str, Any]]]
"""A message's content: either plain text (for text-only models) or a list of
content-part dicts (for multimodal models)."""


def content_to_str(content: Content) -> str:
    """Extract a plain-text representation from any Content.

    For multimodal content, concatenate all text parts with a space separator.
    Non-text parts get a placeholder description.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        part = parse_multimodal_content(item)
        if isinstance(part, TextPart):
            parts.append(part.text)
        elif isinstance(part, (ImageUrlPart, ImageBase64Part)):
            parts.append(str(part))
        elif isinstance(part, AudioPart):
            parts.append(str(part))
        elif isinstance(part, FilePart):
            parts.append(str(part))
    return " ".join(parts)


def content_has_text(content: Content) -> bool:
    """Check if content includes any text parts."""
    if isinstance(content, str):
        return bool(content)
    if not content:
        return False
    return any(p.get("type") == "text" for p in content)


def content_has_images(content: Content) -> bool:
    """Check if content includes any image parts."""
    if isinstance(content, str):
        return False
    if not content:
        return False
    return any(p.get("type") in ("image_url", "image", "image_base64") for p in content)


def split_text_and_images(content: list[dict[str, Any]]) -> tuple[list[TextPart], list[dict[str, Any]]]:
    """Split a mixed content list into (text_parts, image_dicts)."""
    texts: list[TextPart] = []
    images: list[dict[str, Any]] = []
    for item in content:
        if item.get("type") == "text":
            texts.append(TextPart.from_dict(item))
        elif item.get("type") in ("image_url", "image", "image_base64"):
            images.append(item)
    return texts, images


def content_to_openai_blocks(content: Content) -> list[dict[str, Any]]:
    """Convert any Content to OpenAI-compatible content blocks.

    This is used by OpenAIProvider and all OpenAI-compatible providers.
    """
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not content:
        return [{"type": "text", "text": ""}]
    result: list[dict[str, Any]] = []
    for item in content:
        t = item.get("type", "")
        if t == "image_base64":
            # Convert to standard image_url format
            img = ImageBase64Part.from_dict(item)
            result.append(img.to_dict())
        elif t == "audio":
            # Typically only Gemini natively supports audio; for OpenAI we
            # embed the transcript as text if available.
            audio = AudioPart.from_dict(item)
            if audio.transcript:
                result.append({"type": "text", "text": f"[Audio transcript]: {audio.transcript}"})
            else:
                result.append({"type": "text", "text": f"[Audio: {audio.format}, {len(audio.data)} bytes]"})
        elif t == "file":
            f = FilePart.from_dict(item)
            result.append({"type": "text", "text": f"[File: {f.file_name} ({f.mime_type})]"})
        else:
            result.append(item)
    return result


def parse_multimodal_content(raw: dict[str, Any]) -> ContentPart:
    """Parse a single content dict into a typed ContentPart."""
    t = raw.get("type", "")
    if t == "text":
        return TextPart.from_dict(raw)
    if t in ("image_url", "image"):
        url_val = raw.get("image_url", {})
        if isinstance(url_val, str):
            return ImageUrlPart(url=url_val)
        url_str = url_val.get("url", "")
        if url_str.startswith("data:"):
            return ImageBase64Part.from_dict(raw)
        return ImageUrlPart.from_dict(raw)
    if t == "image_base64":
        return ImageBase64Part.from_dict(raw)
    if t == "audio":
        return AudioPart.from_dict(raw)
    if t == "file":
        return FilePart.from_dict(raw)
    # Unknown type — treat as text
    return TextPart(text=str(raw))


def sanitize_content(content: Content) -> Content:
    """Normalize content: convert str to list of blocks if needed for multimodal.

    Returns the content unchanged if it's already a list or if it's a plain string.
    This is a no-op for most cases; providers decide how to route.
    """
    return content


# ---------------------------------------------------------------------------
# Convenience: auto-detect part type from a source string
# ---------------------------------------------------------------------------


def source_to_part(source: str | Path) -> ContentPart:
    """Auto-detect content part type from a file path, URL, or data URI.

    Args:
        source: A file path, http(s) URL, or data URI.

    Returns:
        The appropriate ContentPart.
    """
    if isinstance(source, Path):
        source = str(source)

    # Data URI
    if source.startswith("data:"):
        header, b64 = source.split(",", 1)
        media_type = "image/jpeg"
        if ":" in header.split(";")[0]:
            media_type = header.split(":")[1].split(";")[0]
        if media_type.startswith("image/"):
            return ImageBase64Part(data=b64, media_type=media_type)
        if media_type.startswith("audio/"):
            fmt = media_type.split("/")[1]
            return AudioPart(data=b64, format=fmt)
        return FilePart(file_name="data", data=b64, mime_type=media_type)

    # HTTP(S) URL → image (cannot know without fetching)
    if source.startswith(("http://", "https://")):
        return ImageUrlPart(url=source)

    # Local file path
    path = Path(source)
    if not path.exists():
        # Assume it's text
        return TextPart(text=source)

    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        b64_data = base64.b64encode(path.read_bytes()).decode("ascii")
        if mime.startswith("image/"):
            return ImageBase64Part(data=b64_data, media_type=mime)
        if mime.startswith("audio/"):
            fmt = mime.split("/")[1]
            return AudioPart(data=b64_data, format=fmt)
        return FilePart(file_name=path.name, data=b64_data, mime_type=mime)

    # Fallback: plain text
    return TextPart(text=source)
