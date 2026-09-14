"""Tests for morainet.memory.document_parser — parsing and chunking."""
from __future__ import annotations

import pytest

from morainet.memory.document_parser import (
    CSVParser,
    DocumentLoader,
    MarkdownParser,
    ParsedDocument,
    TextChunker,
    _get_parser,
    register_parser,
)
from morainet.memory.embeddings import HashEmbedder
from morainet.memory.stores import InMemoryVectorStore


def test_text_chunker_fixed(tmp_path):
    chunker = TextChunker(chunk_size=10, chunk_overlap=2, mode="fixed")
    chunks = chunker.split("a" * 25)
    assert len(chunks) >= 3
    # overlap preserved across chunk boundary
    assert chunks[0][-2:] == chunks[1][:2]


def test_text_chunker_recursive():
    chunker = TextChunker(chunk_size=20, chunk_overlap=0, mode="recursive")
    text = "First sentence. Second sentence. Third sentence here."
    chunks = chunker.split(text)
    assert len(chunks) >= 1
    assert all(len(c) <= 20 for c in chunks)


def test_text_chunker_empty():
    chunker = TextChunker()
    assert chunker.split("") == []


def test_markdown_parser_strips_frontmatter(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("---\ntitle: Hello\n---\n\n# Heading\n\nBody text.\n", encoding="utf-8")
    text = MarkdownParser().parse(p)
    assert "title: Hello" not in text
    assert "Body text." in text


def test_csv_parser(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,age\nalice,30\nbob,25\n", encoding="utf-8")
    text = CSVParser().parse(p)
    assert "name" in text
    assert "alice" in text


def test_get_parser_unknown_extension():
    with pytest.raises(ValueError, match="No parser"):
        _get_parser(".xyz")


def test_register_parser_custom():
    class MyParser(MarkdownParser):
        pass

    register_parser(".mypy", MyParser)
    assert _get_parser(".mypy") is not None


def test_document_loader_load_file_md(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome content here for testing.\n", encoding="utf-8")
    loader = DocumentLoader(chunk_size=50, chunk_overlap=0)
    doc = loader.load_file(p)
    assert isinstance(doc, ParsedDocument)
    assert doc.chunks
    assert doc.title == "doc"


async def test_document_loader_build_knowledge_base(tmp_path):
    d = tmp_path / "kb"
    d.mkdir()
    (d / "a.md").write_text("Morainet agent runtime framework documentation.\n", encoding="utf-8")
    (d / "b.md").write_text("RAG pipeline retrieval augmented generation notes.\n", encoding="utf-8")
    store = InMemoryVectorStore()
    loader = DocumentLoader(chunk_size=50, chunk_overlap=0, embedder=HashEmbedder())
    count = await loader.build_knowledge_base(d, store=store)
    assert count == 2
    assert await store.count() == 2
