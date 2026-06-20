"""Provider adapters for multimodal content translation.

Each LLM vendor has a different multimodal format. This module provides a
unified adapter that translates Morainet's vendor-neutral ``Content`` format
to each vendor's native representation.

Supported vendors / image handling:
- **OpenAI / GPT-4V** — ``image_url`` content parts (URL or data-URI)
- **Anthropic / Claude 3 Vision** — ``content`` blocks with ``source``
- **Google / Gemini** — ``inlineData`` parts
- **Ollama** — ``images`` field (base64 list)

Usage::

    from morainet.multimodal import default_adapter

    openai_msgs = default_adapter.to_openai(messages)
    anthropic_msgs = default_adapter.to_anthropic(messages)
    ollama_msgs = default_adapter.to_ollama(messages)
    gemini_content = default_adapter.to_gemini(messages)
"""

from __future__ import annotations

from typing import Any

from morainet.core.models import Message, Role


# ---------------------------------------------------------------------------
# Content block type detection
# ---------------------------------------------------------------------------


def _is_multimodal(content: Any) -> bool:
    """Check if content is a multimodal block list (not plain str)."""
    return isinstance(content, list) and len(content) > 0


def _extract_images(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract image parts from a content list."""
    images: list[dict[str, Any]] = []
    for item in content:
        if item.get("type") in ("image_url", "image"):
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict):
                images.append(image_url)
            else:
                images.append({"url": str(image_url)})
    return images


def _extract_text(content: list[dict[str, Any]]) -> str:
    """Extract concatenated text from a content list."""
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        elif item.get("type") in ("image_url", "image"):
            parts.append("[Image]")
        elif item.get("type") == "audio":
            a = item.get("audio", {})
            parts.append(a.get("transcript", "[Audio]"))
        elif item.get("type") == "file":
            f = item.get("file", {})
            parts.append(f"[File: {f.get('file_name', 'unknown')}]")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Multimodal adapter — registry pattern
# ---------------------------------------------------------------------------


class MultimodalAdapter:
    """Translate Morainet Messages to vendor-specific multimodal formats.

    This is the central switchboard. Each ``to_*`` method inspects
    ``Message.content`` and routes plain text directly or builds rich
    multimodal blocks for the target vendor.

    Extend by registering custom adapters::

        adapter.register("my_vendor", lambda msg: ...)
    """

    def __init__(self) -> None:
        self._custom: dict[str, Any] = {}

    def register(self, vendor: str, adapter_fn: Any) -> None:
        """Register a custom vendor adapter function.

        Adapter receives ``(messages: list[Message])`` and returns
        ``list[dict]``.
        """
        self._custom[vendor] = adapter_fn

    # -- OpenAI / GPT-4V ------------------------------------------------

    def to_openai(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to OpenAI Chat Completions format.

        Text-only messages → ``{"role": ..., "content": "..."}``
        Multimodal messages → ``{"role": ..., "content": [...]}``
        """
        result: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role.value}

            if m.role == Role.TOOL:
                msg["content"] = m.content or ""
                msg["tool_call_id"] = m.tool_call_id
            elif _is_multimodal(m.content) and m.role == Role.USER:
                # Build OpenAI multimodal content array
                blocks: list[dict[str, Any]] = []
                for item in m.content:  # type: ignore[union-attr]
                    if item.get("type") == "image_base64":  # type: ignore[union-attr]
                        # Normalize to standard image_url + data URI
                        b64 = item.get("data", "")  # type: ignore[union-attr]
                        media = item.get("media_type", "image/jpeg")  # type: ignore[union-attr]
                        data_uri = f"data:{media};base64,{b64}"
                        blocks.append({
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": item.get("detail", "auto")},  # type: ignore[union-attr]
                        })
                    elif item.get("type") == "audio":  # type: ignore[union-attr]
                        # OpenAI doesn't support native audio input blocks
                        audio = item.get("audio", {})  # type: ignore[union-attr]
                        transcript = audio.get("transcript", "")
                        if transcript:
                            blocks.append({"type": "text", "text": f"[Audio transcript]: {transcript}"})
                        else:
                            blocks.append({"type": "text", "text": f"[Audio: {audio.get('format', 'unknown')}]"})
                    elif item.get("type") == "file":  # type: ignore[union-attr]
                        f = item.get("file", {})  # type: ignore[union-attr]
                        blocks.append({"type": "text", "text": f"[File: {f.get('file_name', 'unknown')}]"})
                    else:
                        blocks.append(item)  # type: ignore[arg-type]
                msg["content"] = blocks if blocks else ""
            else:
                msg["content"] = m.content or ""

            if m.tool_calls:
                import json
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            result.append(msg)
        return result

    # -- Anthropic / Claude 3 Vision -----------------------------------

    def to_anthropic(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Anthropic Messages API format.

        Returns (system_prompt, converted_messages).
        Image parts become ``source`` blocks with base64 data.
        """
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for m in messages:
            if m.role == Role.SYSTEM:
                if m.content:
                    system_parts.append(
                        m.content if isinstance(m.content, str)
                        else _extract_text(m.content)
                    )
                continue

            if m.role == Role.TOOL:
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }],
                })
                continue

            if m.role == Role.ASSISTANT and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append(
                        {"type": "text", "text": m.content if isinstance(m.content, str) else ""}
                    )
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                converted.append({"role": "assistant", "content": blocks})
                continue

            # User / plain assistant
            if _is_multimodal(m.content) and m.role == Role.USER:
                blocks = self._build_anthropic_blocks(m.content)  # type: ignore[arg-type]
                converted.append({"role": "user", "content": blocks})
            else:
                text = m.content if isinstance(m.content, str) else _extract_text(m.content or [])
                converted.append({"role": m.role.value, "content": text or ""})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, converted

    def _build_anthropic_blocks(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build Anthropic content blocks from a multimodal content list."""
        blocks: list[dict[str, Any]] = []

        for item in content:
            t = item.get("type", "text")
            if t == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif t in ("image_url", "image"):
                img = item.get("image_url", {})
                if isinstance(img, dict):
                    url = img.get("url", "")
                else:
                    url = str(img)

                if url.startswith("data:"):
                    # data:image/png;base64,...
                    header, b64 = url.split(",", 1)
                    media_type = "image/jpeg"
                    if ":" in header.split(";")[0]:
                        media_type = header.split(":")[1].split(";")[0]
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })

            elif t == "image_base64":
                b64 = item.get("data", "")
                media = item.get("media_type", "image/jpeg")
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": b64,
                    },
                })
            elif t == "audio":
                audio = item.get("audio", {})
                transcript = audio.get("transcript", "")
                if transcript:
                    blocks.append({"type": "text", "text": f"[Audio transcript]: {transcript}"})
                else:
                    blocks.append({"type": "text", "text": f"[Audio: {audio.get('format', 'unknown')}]"})
            elif t == "file":
                f = item.get("file", {})
                blocks.append({"type": "text", "text": f"[File: {f.get('file_name', 'unknown')}]"})
            else:
                blocks.append({"type": "text", "text": str(item)})
        return blocks

    # -- Google / Gemini --------------------------------------------------

    def to_gemini(
        self, messages: list[Message]
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Convert messages to Gemini GenerateContent format.

        Returns (system_instruction, contents).
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for m in messages:
            if m.role == Role.SYSTEM:
                if m.content:
                    system_parts.append(
                        m.content if isinstance(m.content, str)
                        else _extract_text(m.content)
                    )
                continue

            if m.role == Role.TOOL:
                contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": m.tool_call_id or "",
                            "response": {"result": m.content or ""},
                        }
                    }],
                })
                continue

            if m.role == Role.ASSISTANT and m.tool_calls:
                parts: list[dict[str, Any]] = []
                if m.content:
                    parts.append({"text": m.content if isinstance(m.content, str) else ""})
                for tc in m.tool_calls:
                    parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
                contents.append({"role": "model", "parts": parts})
                continue

            # User / plain assistant — build parts
            role_str = "model" if m.role == Role.ASSISTANT else "user"

            if _is_multimodal(m.content) and m.role == Role.USER:
                parts = self._build_gemini_parts(m.content)  # type: ignore[arg-type]
                contents.append({"role": role_str, "parts": parts})
            else:
                text = m.content if isinstance(m.content, str) else _extract_text(m.content or [])
                contents.append({"role": role_str, "parts": [{"text": text or ""}]})

        system = (
            {"parts": [{"text": "\n\n".join(system_parts)}]}
            if system_parts else None
        )
        return system, contents

    def _build_gemini_parts(self, content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build Gemini parts from a multimodal content list."""
        parts: list[dict[str, Any]] = []

        for item in content:
            t = item.get("type", "text")
            if t == "text":
                parts.append({"text": item.get("text", "")})
            elif t in ("image_url", "image"):
                img = item.get("image_url", {})
                if isinstance(img, dict):
                    url = img.get("url", "")
                else:
                    url = str(img)

                if url.startswith("data:"):
                    header, b64 = url.split(",", 1)
                    media_type = "image/jpeg"
                    if ":" in header.split(";")[0]:
                        media_type = header.split(":")[1].split(";")[0]
                    parts.append({
                        "inlineData": {"mimeType": media_type, "data": b64},
                    })
                else:
                    # URL → Gemini needs data; fetch or pass through
                    parts.append({
                        "fileData": {"fileUri": url},
                    })

            elif t == "image_base64":
                parts.append({
                    "inlineData": {
                        "mimeType": item.get("media_type", "image/jpeg"),
                        "data": item.get("data", ""),
                    },
                })
            elif t == "audio":
                audio = item.get("audio", {})
                parts.append({
                    "inlineData": {
                        "mimeType": f"audio/{audio.get('format', 'mp3')}",
                        "data": audio.get("data", ""),
                    },
                })
            elif t == "file":
                f = item.get("file", {})
                parts.append({
                    "inlineData": {
                        "mimeType": f.get("mime_type", "application/octet-stream"),
                        "data": f.get("data", ""),
                    },
                })
            else:
                parts.append({"text": str(item)})
        return parts

    # -- Ollama ------------------------------------------------------------

    def to_ollama(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages to Ollama /api/chat format.

        Ollama supports an ``images`` field on user messages (list of base64).
        """
        converted: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role.value}

            if _is_multimodal(m.content) and m.role == Role.USER:
                text = _extract_text(m.content)  # type: ignore[arg-type]
                msg["content"] = text
                # Collect images
                ollama_images: list[str] = []
                for item in m.content:  # type: ignore[union-attr]
                    if item.get("type") in ("image_url", "image"):  # type: ignore[union-attr,operator]  # type: ignore[union-attr]
                        img = item.get("image_url", {})  # type: ignore[union-attr]
                        if isinstance(img, dict):
                            url = img.get("url", "")
                        else:
                            url = str(img)
                        if url.startswith("data:image/"):
                            _, b64 = url.split(",", 1)
                            ollama_images.append(b64)
                    elif item.get("type") == "image_base64":  # type: ignore[union-attr]
                        ollama_images.append(item.get("data", ""))  # type: ignore[union-attr]
                if ollama_images:
                    msg["images"] = ollama_images
            else:
                msg["content"] = m.content if isinstance(m.content, str) else (
                    _extract_text(m.content or []) if m.content else ""
                )

            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
            converted.append(msg)
        return converted

    # -- Generic (best-effort) ---------------------------------------------

    def to_generic(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert to a best-effort generic format.

        Images are desugared to text placeholders; text is preserved.
        Suitable for text-only models.
        """
        converted: list[dict[str, Any]] = []
        for m in messages:
            msg: dict[str, Any] = {"role": m.role.value}
            if m.content is None:
                msg["content"] = ""
            elif isinstance(m.content, str):
                msg["content"] = m.content
            else:
                msg["content"] = _extract_text(m.content)
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in m.tool_calls
                ]
            if m.tool_call_id is not None:
                msg["tool_call_id"] = m.tool_call_id
            converted.append(msg)
        return converted


# ---------------------------------------------------------------------------
# Default instance
# ---------------------------------------------------------------------------


default_adapter = MultimodalAdapter()
"""The default multimodal adapter instance used throughout the framework."""


def get_adapter(vendor: str) -> "MultimodalAdapter":
    """Get a configured adapter for a specific vendor.

    Args:
        vendor: ``"openai"``, ``"anthropic"``, ``"gemini"``, ``"ollama"``.

    Returns a MultimodalAdapter. The same adapter handles all vendors.
    """
    return default_adapter


def register_adapter(vendor: str, adapter_fn: Any) -> None:
    """Register a custom vendor adapter on the default instance."""
    default_adapter.register(vendor, adapter_fn)
