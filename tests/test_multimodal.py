"""Tests for the morainet.multimodal package.

Covers content parts + utilities, provider adapters, multimodal RAG, and the
built-in multimodal tools. All tests are hermetic: no real vision/LLM API calls
and no images downloaded from the network. Tiny in-memory images and stub
callables/providers are used throughout.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from morainet.core.models import ChatResponse, Message, Usage
from morainet.multimodal import (
    AudioPart,
    ContentType,
    FilePart,
    ImageBase64Part,
    ImageUrlPart,
    TextPart,
    chart_parse,
    content_has_images,
    content_has_text,
    content_to_openai_blocks,
    content_to_str,
    describe_image,
    image_understand,
    ocr,
    parse_multimodal_content,
    sanitize_content,
    source_to_part,
    speech_to_text,
    split_text_and_images,
)
from morainet.multimodal.provider_adapter import (
    MultimodalAdapter,
    default_adapter,
    get_adapter,
    register_adapter,
)
from morainet.multimodal.rag import (
    MultimodalDocument,
    MultimodalRAG,
    MultimodalRetriever,
    SimpleImageCaptioner,
    SimpleImageTextEncoder,
    VisionReasoningChain,
)
from morainet.multimodal.tools import ImageAnalysisResult, MultimodalToolRunner
from morainet.providers.mock import MockProvider


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


TINY_PNG_B64 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 20 + b"tinyfake"
).decode("ascii")


@pytest.fixture
def tiny_image_bytes() -> bytes:
    """A tiny in-memory image blob (no real image needed)."""
    return base64.b64decode(TINY_PNG_B64)


def make_response(text: str) -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(content=text),
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="mock",
        finish_reason="stop",
    )


class StubEmbedder:
    """Deterministic word-overlap text embedder (bag-of-words style).

    Cosine/dot-product similarity reflects shared tokens, so retrieval ordering
    is meaningful and reproducible in tests.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        vec = [0.0] * self.dim
        for tok in text.lower().replace(",", " ").replace(".", " ").split():
            if not tok:
                continue
            h = 0
            for ch in tok:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            vec[h % self.dim] = 1.0
        return vec


class StubStore:
    """In-memory vector store for hermetic RAG tests."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    async def upsert(self, text: str, embedding: list[float], meta: dict) -> None:
        self.records.append({"text": text, "embedding": embedding, "meta": meta})

    async def search(self, embedding: list[float], top_k: int = 5) -> list[dict]:
        scored = []
        for r in self.records:
            score = sum(a * b for a, b in zip(embedding, r["embedding"]))
            scored.append({"id": r["meta"]["id"], "text": r["text"], "score": score, "meta": r["meta"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class FakeCaptioner:
    """Captioner stub used to avoid real vision calls."""

    async def caption(self, image_source) -> str:
        return "a picture of a cat"


# ---------------------------------------------------------------------------
# Content parts — validation + serialization
# ---------------------------------------------------------------------------


def test_text_part_roundtrip():
    part = TextPart(text="hello")
    assert part.type == ContentType.TEXT
    assert part.to_dict() == {"type": "text", "text": "hello"}
    assert TextPart.from_dict(part.to_dict()) == part
    assert str(part) == "hello"


def test_image_url_part_roundtrip():
    part = ImageUrlPart(url="https://example.com/a.png", detail="high")
    assert part.type == ContentType.IMAGE_URL
    assert part.to_dict() == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png", "detail": "high"},
    }
    assert ImageUrlPart.from_dict(part.to_dict()) == part
    assert "example.com" in str(part)

    # str form of image_url
    assert ImageUrlPart.from_dict({"type": "image_url", "image_url": "https://x/y.png"}).url == "https://x/y.png"


def test_image_base64_part_roundtrip():
    part = ImageBase64Part(data="AAAA", media_type="image/png")
    assert part.type == ContentType.IMAGE_BASE64
    d = part.to_dict()
    assert d["type"] == "image_url"
    assert d["image_url"]["url"] == "data:image/png;base64,AAAA"
    assert part.to_data_uri() == "data:image/png;base64,AAAA"
    restored = ImageBase64Part.from_dict(d)
    assert restored.data == "AAAA"
    assert restored.media_type == "image/png"
    assert "bytes" in str(part)


def test_audio_part_roundtrip():
    part = AudioPart(data="AAAA", format="mp3", transcript="hello world")
    assert part.type == ContentType.AUDIO
    d = part.to_dict()
    assert d["type"] == "audio"
    assert d["audio"]["data"] == "AAAA"
    assert d["audio"]["transcript"] == "hello world"
    assert AudioPart.from_dict(d) == part
    assert "hello world" in str(part)


def test_file_part_roundtrip():
    part = FilePart(file_name="report.pdf", data="AA==", mime_type="application/pdf")
    assert part.type == ContentType.FILE
    d = part.to_dict()
    assert d["type"] == "file"
    assert d["file"]["file_name"] == "report.pdf"
    assert FilePart.from_dict(d) == part
    assert "report.pdf" in str(part)


# ---------------------------------------------------------------------------
# Content utilities
# ---------------------------------------------------------------------------


def test_content_to_str_variants():
    assert content_to_str(None) == ""
    assert content_to_str("plain") == "plain"
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
        {"type": "image_base64", "data": "AA==", "media_type": "image/png"},
        {"type": "audio", "audio": {"data": "AA==", "format": "mp3", "transcript": "hi"}},
        {"type": "file", "file": {"file_name": "f.txt", "data": "AA=="}},
    ]
    out = content_to_str(content)
    assert "hello" in out
    assert "hi" in out  # audio transcript
    assert "f.txt" in out  # file name


def test_content_has_text_and_images():
    assert content_has_text("hello") is True
    assert content_has_text("") is False
    assert content_has_text([{"type": "text", "text": "x"}]) is True
    assert content_has_text([{"type": "image_url", "image_url": {"url": "u"}}]) is False

    assert content_has_images("str") is False
    assert content_has_images([]) is False
    assert content_has_images([{"type": "image_url", "image_url": {"url": "u"}}]) is True
    assert content_has_images([{"type": "image", "image_url": {"url": "u"}}]) is True
    assert content_has_images([{"type": "image_base64", "data": "AA=="}]) is True
    assert content_has_images([{"type": "text", "text": "x"}]) is False


def test_split_text_and_images():
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "u1"}},
        {"type": "image", "image_url": {"url": "u2"}},
        {"type": "image_base64", "data": "AA=="},
        {"type": "audio", "audio": {"data": "AA=="}},
    ]
    texts, images = split_text_and_images(content)
    assert len(texts) == 1
    assert texts[0].text == "hello"
    assert len(images) == 3


def test_content_to_openai_blocks():
    # text only
    assert content_to_openai_blocks(None) == [{"type": "text", "text": ""}]
    assert content_to_openai_blocks("hi") == [{"type": "text", "text": "hi"}]
    assert content_to_openai_blocks([]) == [{"type": "text", "text": ""}]

    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_base64", "data": "AA==", "media_type": "image/png"},
        {"type": "audio", "audio": {"data": "AA==", "format": "mp3", "transcript": "hi"}},
        {"type": "audio", "audio": {"data": "AA==", "format": "wav"}},
        {"type": "file", "file": {"file_name": "f.txt", "data": "AA=="}},
        {"type": "image_url", "image_url": {"url": "https://x/a.png"}},
    ]
    blocks = content_to_openai_blocks(content)
    assert blocks[0] == {"type": "text", "text": "hello"}
    # image_base64 normalizes to an image_url block with a data-URI URL
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:")
    assert "hi" in blocks[2]["text"]
    assert "wav" in blocks[3]["text"]
    assert "f.txt" in blocks[4]["text"]
    assert blocks[5] == {"type": "image_url", "image_url": {"url": "https://x/a.png"}}


def test_parse_multimodal_content_unknown_type_treated_as_text():
    part = parse_multimodal_content({"type": "weird", "foo": "bar"})
    assert isinstance(part, TextPart)


# ---------------------------------------------------------------------------
# sanitize_content / source_to_part
# ---------------------------------------------------------------------------


def test_sanitize_content_is_identity():
    assert sanitize_content("hello") == "hello"
    content = [{"type": "text", "text": "x"}]
    assert sanitize_content(content) == content


def test_source_to_part_url_and_data_uri():
    url_part = source_to_part("https://example.com/photo.png")
    assert isinstance(url_part, ImageUrlPart)
    assert url_part.url == "https://example.com/photo.png"

    data_uri = f"data:image/png;base64,{TINY_PNG_B64}"
    b64_part = source_to_part(data_uri)
    assert isinstance(b64_part, ImageBase64Part)
    assert b64_part.media_type == "image/png"
    assert b64_part.data == TINY_PNG_B64

    audio_uri = f"data:audio/mp3;base64,{TINY_PNG_B64}"
    audio_part = source_to_part(audio_uri)
    assert isinstance(audio_part, AudioPart)
    assert audio_part.format == "mp3"

    file_uri = f"data:application/pdf;base64,{TINY_PNG_B64}"
    file_part = source_to_part(file_uri)
    assert isinstance(file_part, FilePart)
    assert file_part.mime_type == "application/pdf"


def test_source_to_part_local_file(tmp_path: Path):
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    part = source_to_part(img)
    assert isinstance(part, ImageBase64Part)
    assert part.media_type == "image/png"
    # Path object input
    assert isinstance(source_to_part(Path(img)), ImageBase64Part)

    audio = tmp_path / "note.wav"
    audio.write_bytes(b"RIFFfake")
    assert isinstance(source_to_part(audio), AudioPart)

    txt = tmp_path / "notes.txt"
    txt.write_text("some notes", encoding="utf-8")
    filepart = source_to_part(txt)
    assert isinstance(filepart, FilePart)
    assert filepart.mime_type == "text/plain"

    # nonexistent path → text fallback
    missing = source_to_part(str(tmp_path / "nope.txt"))
    assert isinstance(missing, TextPart)
    assert missing.text == str(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# Provider adapter
# ---------------------------------------------------------------------------


def test_multimodal_adapter_to_openai():
    adapter = MultimodalAdapter()
    msgs = [
        Message.system("be brief"),
        Message.user(
            [
                {"type": "text", "text": "What's this?"},
                {"type": "image_base64", "data": "AA==", "media_type": "image/png"},
            ]
        ),
        Message.assistant(content="a dog"),
    ]
    out = adapter.to_openai(msgs)
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1]["role"] == "user"
    blocks = out[1]["content"]
    assert blocks[0] == {"type": "text", "text": "What's this?"}
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"] == "data:image/png;base64,AA=="
    assert out[2] == {"role": "assistant", "content": "a dog"}


def test_multimodal_adapter_to_anthropic():
    adapter = MultimodalAdapter()
    msgs = [
        Message.system("be brief"),
        Message.user(
            [
                {"type": "text", "text": "hi"},
                {"type": "image_base64", "data": "AA==", "media_type": "image/png"},
            ]
        ),
    ]
    system, out = adapter.to_anthropic(msgs)
    assert system == "be brief"
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["type"] == "base64"
    assert blocks[1]["source"]["data"] == "AA=="


def test_multimodal_adapter_to_gemini_and_ollama():
    adapter = MultimodalAdapter()
    msgs = [
        Message.system("be brief"),
        Message.user(
            [
                {"type": "text", "text": "hi"},
                {"type": "image_base64", "data": "AA==", "media_type": "image/png"},
            ]
        ),
    ]
    system, contents = adapter.to_gemini(msgs)
    assert system == {"parts": [{"text": "be brief"}]}
    parts = contents[0]["parts"]
    assert {"text": "hi"} in parts
    assert any("inlineData" in p and p["inlineData"]["data"] == "AA==" for p in parts)

    ollama = adapter.to_ollama(msgs)
    # [0] is the system message, [1] is the user message
    user = ollama[1]
    assert user["role"] == "user"
    assert user["content"] == "hi"
    assert user["images"] == ["AA=="]


def test_adapter_custom_register_and_defaults():
    adapter = MultimodalAdapter()
    adapter.register("myvendor", lambda msgs: [{"role": m.role.value} for m in msgs])
    assert adapter._custom["myvendor"] is not None

    # default_adapter + get_adapter return the same shared instance
    assert get_adapter("openai") is default_adapter
    assert get_adapter("any-vendor") is default_adapter


def test_register_adapter_via_module_function():
    sentinel = []

    def fake_adapter(msgs):
        sentinel.append(msgs)
        return [{"role": "user", "content": "custom"}]

    register_adapter("testvendor", fake_adapter)
    try:
        assert "testvendor" in default_adapter._custom
        assert default_adapter._custom["testvendor"] is fake_adapter
    finally:
        # clean up to avoid leaking state
        del default_adapter._custom["testvendor"]


# ---------------------------------------------------------------------------
# Multimodal RAG
# ---------------------------------------------------------------------------


def test_multimodal_document_text_for_indexing():
    doc = MultimodalDocument(
        id="d1",
        text="hello",
        images=[{"path": "a.png", "caption": "a cat"}],
    )
    idx = doc.text_for_indexing
    assert "hello" in idx
    assert "a cat" in idx


@pytest.mark.asyncio
async def test_simple_image_captioner_with_mock_provider(tiny_image_bytes):
    provider = MockProvider()
    captioner = SimpleImageCaptioner(provider)
    caption = await captioner.caption(tiny_image_bytes)
    assert "[mock]" in caption

    # URL path
    caption_url = await captioner.caption("https://example.com/photo.png")
    assert "[mock]" in caption_url


@pytest.mark.asyncio
async def test_simple_image_text_encoder():
    embedder = StubEmbedder()
    encoder = SimpleImageTextEncoder(embedder=embedder, captioner=FakeCaptioner())

    text_vec = await encoder.embed_text("hello")
    assert len(text_vec) == 32

    img_vec = await encoder.embed_image("whatever.png")
    assert len(img_vec) == 32

    multi_vec = await encoder.embed_multimodal("hello", "whatever.png")
    assert len(multi_vec) == 32

    # Without captioner, image embedding raises
    encoder_no_cap = SimpleImageTextEncoder(embedder=embedder, captioner=None)
    with pytest.raises(RuntimeError):
        await encoder_no_cap.embed_image("whatever.png")

    # embed_multimodal with no image just uses text
    only_text = await encoder_no_cap.embed_multimodal("hello")
    assert len(only_text) == 32


@pytest.mark.asyncio
async def test_multimodal_retriever_index_and_search():
    embedder = StubEmbedder()
    store = StubStore()
    retriever = MultimodalRetriever(store=store, embedder=embedder, captioner=FakeCaptioner())

    docs = [
        MultimodalDocument(id="d1", text="the cat sits on the mat", images=[{"path": "c.png", "caption": "a cat"}]),
        MultimodalDocument(id="d2", text="the dog runs fast"),
    ]
    count = await retriever.index(docs)
    assert count == 2

    hits = await retriever.search("cat mat", top_k=2)
    assert len(hits) == 2
    # d1 should be the top hit for "cat mat"
    assert hits[0]["id"] == "d1"
    assert hits[0]["images"] == [{"path": "c.png", "caption": "a cat"}]


@pytest.mark.asyncio
async def test_multimodal_rag_manual_ingest_and_query():
    embedder = StubEmbedder()
    store = StubStore()
    rag = MultimodalRAG(store=store, embedder=embedder, captioner=FakeCaptioner())

    count = await rag.ingest_manual(
        texts=["architecture diagram shows three layers", "unrelated note about weather"],
        images_per_doc=[["arch.png"], None],
        metadata_list=[{"source": "docs/arch.txt"}, {}],
    )
    assert count == 2

    results = await rag.query("architecture layers", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] != ""
    assert results[0]["metadata"]["source"] == "docs/arch.txt"
    # The doc with the image has one image attached
    assert results[0]["images"]


@pytest.mark.asyncio
async def test_multimodal_rag_ingest_directory(tmp_path: Path):
    embedder = StubEmbedder()
    store = StubStore()
    rag = MultimodalRAG(store=store, embedder=embedder, captioner=FakeCaptioner())

    text_dir = tmp_path / "docs"
    text_dir.mkdir()
    (text_dir / "readme.md").write_text("System architecture overview\n" * 60, encoding="utf-8")
    (text_dir / "junk.txt").write_bytes(b"\xff\xfe\x00\x01")  # undecodable -> skipped

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "readme.png").write_bytes(b"\x89PNGfake")

    count = await rag.ingest_directory(text_dir, image_dir)
    assert count >= 1

    # chunked doc should have at least one associated image (by stem match)
    results = await rag.query("architecture", top_k=5)
    assert any(r["images"] for r in results)


@pytest.mark.asyncio
async def test_vision_reasoning_chain():
    provider = MockProvider(handler=lambda msgs, tools: make_response("synthesized answer"))
    embedder = StubEmbedder()
    store = StubStore()
    rag = MultimodalRAG(store=store, embedder=embedder, captioner=FakeCaptioner())

    # index a doc with an image file so vision step reads a real path
    tmp = Path(__file__).parent / "_tmp_mm_viz.png"
    try:
        tmp.write_bytes(b"\x89PNGfake")
        count = await rag.ingest_manual(
            texts=["the architecture diagram shows three service layers"],
            images_per_doc=[[str(tmp)]],
        )
        assert count == 1

        chain = VisionReasoningChain(provider=provider, rag=rag, max_images=3)
        result = await chain.reason("explain the architecture")
        assert "answer" in result
        assert result["answer"] == "synthesized answer"
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 1
        assert len(result["vision_results"]) == 1
        assert result["vision_results"][0]["image_path"] == str(tmp)
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_vision_reasoning_chain_handles_image_read_failure(tmp_path: Path):
    provider = MockProvider(handler=lambda msgs, tools: make_response("ok"))
    embedder = StubEmbedder()
    store = StubStore()
    rag = MultimodalRAG(store=store, embedder=embedder, captioner=FakeCaptioner())

    missing = str(tmp_path / "does_not_exist.png")
    await rag.ingest_manual(
        texts=["text with a broken image"],
        images_per_doc=[[missing]],
    )
    chain = VisionReasoningChain(provider=provider, rag=rag, max_images=3)
    result = await chain.reason("anything")
    assert result["vision_results"][0]["analysis"] == "[Analysis failed]"


# ---------------------------------------------------------------------------
# Multimodal tools
# ---------------------------------------------------------------------------


def test_image_understand():
    out = image_understand(image_url="https://example.com/photo.png", instruction="What is this?")
    assert "Vision analysis requested" in out
    assert "https://example.com/photo.png" in out
    assert "What is this?" in out


def test_describe_image():
    out = describe_image(image_url="https://example.com/photo.png")
    assert "Vision analysis requested" in out
    assert "one-paragraph description" in out


def test_ocr():
    out = ocr(image_url="https://example.com/receipt.png", language="en")
    assert "OCR requested" in out
    assert "en" in out


def test_chart_parse():
    out = chart_parse(image_url="https://example.com/chart.png", chart_type="bar")
    assert "Chart parsing requested" in out
    assert "bar" in out


def test_speech_to_text():
    out = speech_to_text(audio_url="https://example.com/a.mp3", format="mp3")
    assert "Speech-to-text requested" in out
    assert "mp3" in out


def test_image_analysis_result_defaults():
    res = ImageAnalysisResult(tool="ocr")
    assert res.image_count == 0
    assert res.observations == []
    assert res.text is None
    assert res.raw == ""


@pytest.mark.asyncio
async def test_multimodal_tool_runner_async():
    provider = MockProvider(handler=lambda msgs, tools: make_response("ANALYZED"))
    runner = MultimodalToolRunner(provider)

    out = await runner.image_understand(image_url="https://example.com/photo.png")
    assert out == "ANALYZED"

    out_ocr = await runner.ocr(image_url="https://example.com/receipt.png")
    assert out_ocr == "ANALYZED"

    out_chart = await runner.chart_parse(image_url="https://example.com/chart.png")
    assert out_chart == "ANALYZED"

    # no image provided -> error string
    err = await runner.image_understand()
    assert "Error" in err
