"""Multimodal RAG: image+text mixed retrieval and joint reasoning.

Demonstrates:
- **MultimodalDocument** — documents with text + associated images
- **ImageCaptioner** — auto-caption images for text-based retrieval
- **MultimodalRAG** — end-to-end: ingest, query, retrieve mixed content
- **VisionReasoningChain** — multi-step joint reasoning over text + images

Offline-safe: uses MockProvider + HashEmbedder. For production, swap to
OpenAIProvider(model="gpt-4o") + OpenAIEmbedder().
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

from morainet import (
    Agent,
    MockProvider,
)
from morainet.memory.embeddings import HashEmbedder
from morainet.memory.stores import InMemoryVectorStore


# --- Setup: create demo multimodal documents ------------------------------


def _create_demo_docs() -> tuple[Path, Path]:
    """Create temporary directories with text docs and image placeholders."""
    root = Path(tempfile.mkdtemp(prefix="morainet_mm_"))
    text_dir = root / "docs"
    image_dir = root / "images"
    text_dir.mkdir()
    image_dir.mkdir()

    # Text documents (referencing images)
    (text_dir / "architecture.md").write_text(
        """# System Architecture

Our system uses a microservices architecture with the following components:

1. **API Gateway** — Entry point for all client requests. Handles auth, rate limiting,
   and request routing. See the architecture diagram for details.

2. **Service Mesh** — Service-to-service communication uses Istio with mutual TLS.
   The data flow is shown in the data flow diagram.

3. **Database Layer** — PostgreSQL for transactional data, Redis for caching,
   and Elasticsearch for full-text search.

4. **Message Queue** — Kafka for async event processing between services.

5. **Monitoring Stack** — Prometheus + Grafana + Loki for metrics, dashboards, and logs.
""",
        encoding="utf-8",
    )

    (text_dir / "data_pipeline.md").write_text(
        """# Data Pipeline Design

## Ingestion
Raw data enters via Kafka topics. The ingestion service validates schema,
deduplicates, and writes to the raw data lake (S3/MinIO).

## Processing
- **Batch Layer** — Spark jobs run hourly for aggregation and feature engineering.
- **Stream Layer** — Flink processes real-time events for dashboards.
- **ML Inference** — Model serving via Triton, triggered by new data events.

## Storage
- Raw: S3 (Parquet format)
- Processed: ClickHouse (OLAP queries)
- Features: Redis (low-latency serving)
""",
        encoding="utf-8",
    )

    (text_dir / "deployment.md").write_text(
        """# Deployment Topology

## Environments
- **Dev**: Single-node K8s (kind), CI/CD via GitHub Actions
- **Staging**: 3-node K8s cluster, mirror of production
- **Production**: Multi-AZ K8s, blue-green deployments

## Networking
See the network topology diagram for VPC/subnet layout and security groups.

## Scaling
- HPA based on CPU/memory for stateless services
- KEDA for Kafka consumer auto-scaling
- Cluster Autoscaler for node pool management
""",
        encoding="utf-8",
    )

    # Create dummy image files (1x1 pixel PNGs)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    # Create images with stems matching the doc stems
    (image_dir / "architecture.png").write_bytes(png_bytes)
    (image_dir / "data_pipeline.png").write_bytes(png_bytes)
    (image_dir / "deployment.png").write_bytes(png_bytes)

    return text_dir, image_dir


# --- Example 1: Multimodal document construction --------------------------


def demo_multimodal_documents():
    """Create and inspect MultimodalDocument objects."""
    print("=" * 60)
    print("1. Multimodal Document Construction")
    print("=" * 60)

    from morainet.multimodal.rag import MultimodalDocument

    # Document with text + image references
    doc = MultimodalDocument(
        id="doc-001",
        text="The system uses a microservices architecture with an API Gateway.",
        images=[
            {
                "path": "/images/architecture.png",
                "url": "https://example.com/architecture.png",
                "caption": "A diagram showing microservices connected via API Gateway",
            },
        ],
        metadata={"source": "architecture.md", "section": "overview"},
    )

    print(f"\n  Document ID: {doc.id}")
    print(f"  Text: {doc.text[:80]}...")
    print(f"  Images: {len(doc.images)}")
    print(f"  Image caption: {doc.images[0]['caption'][:60]}...")
    print(f"\n  Text for indexing (includes captions):")
    print(f"    {doc.text_for_indexing[:120]}...")


# --- Example 2: ImageCaptioner --------------------------------------------


async def demo_image_captioner():
    """Demonstrate image captioning (text bridge for retrieval)."""
    print("\n" + "=" * 60)
    print("2. Image Captioning (MockProvider)")
    print("=" * 60)

    from morainet.multimodal.rag import SimpleImageCaptioner

    captioner = SimpleImageCaptioner(provider=MockProvider())

    # Create a dummy PNG and caption it
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    caption = await captioner.caption(png_bytes)
    print(f"\n  Image caption: {caption[:120]}...")


# --- Example 3: Multimodal RAG pipeline -----------------------------------


async def demo_multimodal_rag():
    """Full multimodal RAG pipeline: ingest, query, retrieve."""
    print("\n" + "=" * 60)
    print("3. Multimodal RAG Pipeline")
    print("=" * 60)

    from morainet.multimodal.rag import MultimodalRAG, SimpleImageCaptioner

    # Create demo docs
    text_dir, image_dir = _create_demo_docs()

    # Build RAG pipeline
    captioner = SimpleImageCaptioner(provider=MockProvider())
    rag = MultimodalRAG(
        store=InMemoryVectorStore(),
        embedder=HashEmbedder(),
        captioner=captioner,
        chunk_size=500,
    )

    # Ingest
    print(f"\n  Ingesting from {text_dir}...")
    count = await rag.ingest_directory(text_dir, image_dir)
    print(f"  Indexed {count} chunks")

    # Query
    print(f"\n  Query: 'What is the database layer?'")
    results = await rag.query("What is the database layer?", top_k=3)
    for i, r in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {r['score']:.3f}")
        print(f"    Text: {r['text'][:100]}...")
        print(f"    Images: {len(r.get('images', []))}")

    # Query about images
    print(f"\n  Query: 'Show me the architecture diagram'")
    results = await rag.query("Show me the architecture diagram", top_k=3)
    for i, r in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {r['score']:.3f}")
        print(f"    Text: {r['text'][:100]}...")
        print(f"    Images: {len(r.get('images', []))}")
        for img in r.get("images", []):
            print(f"      - {img.get('path', '')}: {img.get('caption', '')[:60]}...")


# --- Example 4: Vision reasoning chain ------------------------------------


async def demo_vision_reasoning():
    """Multi-step joint reasoning: retrieve → analyze images → synthesize."""
    print("\n" + "=" * 60)
    print("4. Vision Reasoning Chain")
    print("=" * 60)

    from morainet.multimodal.rag import (
        MultimodalRAG,
        SimpleImageCaptioner,
        VisionReasoningChain,
    )

    text_dir, image_dir = _create_demo_docs()
    captioner = SimpleImageCaptioner(provider=MockProvider())
    rag = MultimodalRAG(
        store=InMemoryVectorStore(),
        embedder=HashEmbedder(),
        captioner=captioner,
    )
    count = await rag.ingest_directory(text_dir, image_dir)
    print(f"\n  Indexed {count} chunks")

    chain = VisionReasoningChain(
        provider=MockProvider(),
        rag=rag,
        max_images=2,
    )

    print(f"\n  Query: 'Explain the system architecture'")
    result = await chain.reason("Explain the system architecture")

    print(f"\n  Answer: {result['answer'][:200]}...")
    print(f"  Retrieved documents: {len(result['retrieved_docs'])}")
    print(f"  Vision analyses: {len(result['vision_results'])}")


# --- Example 5: Agent + MultimodalRAG integration -------------------------


async def demo_agent_with_multimodal_rag():
    """Integrate MultimodalRAG with an Agent for interactive QA."""
    print("\n" + "=" * 60)
    print("5. Agent + MultimodalRAG Integration")
    print("=" * 60)

    from morainet.multimodal.rag import MultimodalRAG, SimpleImageCaptioner

    # Build RAG
    text_dir, image_dir = _create_demo_docs()
    captioner = SimpleImageCaptioner(provider=MockProvider())
    rag = MultimodalRAG(
        store=InMemoryVectorStore(),
        embedder=HashEmbedder(),
        captioner=captioner,
    )
    count = await rag.ingest_directory(text_dir, image_dir)
    print(f"\n  RAG indexed {count} chunks")

    from morainet import tool as mora_tool

    # Expose RAG query as a tool
    @mora_tool
    def search_knowledge_base(query: str) -> str:
        """Search the multimodal knowledge base for relevant documents.

        Args:
            query: The search query string.
        """
        import asyncio

        async def _search():
            results = await rag.query(query, top_k=3)
            lines = []
            for i, r in enumerate(results):
                img_info = f", {len(r.get('images', []))} image(s)" if r.get("images") else ""
                lines.append(
                    f"[Doc {i+1}] (score={r['score']:.3f}{img_info})\n"
                    f"{r['text'][:300]}"
                )
            return "\n\n".join(lines) if lines else "No results."

        return asyncio.run(_search())

    agent = Agent(
        provider=MockProvider(),
        tools=[search_knowledge_base],
        system_prompt=(
            "You are a technical documentation assistant. Use "
            "search_knowledge_base to find relevant documents and answer "
            "questions about the system architecture, data pipeline, and deployment."
        ),
    )

    query = "How does the data pipeline work end-to-end?"
    print(f"\n  Query: {query}")
    result = await agent.arun(query)
    print(f"  Answer: {result.final_answer[:300]}...")


# --- Example 6: Manual multimodal document ingestion ----------------------


async def demo_manual_ingestion():
    """Manually ingest text+image pairs for custom multimodal RAG."""
    print("\n" + "=" * 60)
    print("6. Manual Multimodal Ingestion")
    print("=" * 60)

    from morainet.multimodal.rag import MultimodalRAG, SimpleImageCaptioner

    captioner = SimpleImageCaptioner(provider=MockProvider())

    # Create test images
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    img_path = Path(tempfile.mkdtemp(prefix="morainet_mm2_")) / "test.png"
    img_path.write_bytes(png_bytes)

    rag = MultimodalRAG(
        store=InMemoryVectorStore(),
        embedder=HashEmbedder(),
        captioner=captioner,
    )

    count = await rag.ingest_manual(
        texts=[
            "The frontend uses React with TypeScript. See the component tree diagram.",
            "The backend API is built with FastAPI and uses PostgreSQL. See the ERD diagram.",
            "CI/CD pipeline runs in GitHub Actions with Docker build and K8s deploy.",
        ],
        images_per_doc=[
            [str(img_path)],   # component tree
            [str(img_path)],   # ERD
            None,              # no image
        ],
        metadata_list=[
            {"section": "frontend", "team": "ui"},
            {"section": "backend", "team": "api"},
            {"section": "devops", "team": "infra"},
        ],
    )
    print(f"\n  Manually indexed {count} documents")

    results = await rag.query("database schema", top_k=3)
    for i, r in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {r['score']:.3f}")
        print(f"    Text: {r['text'][:100]}...")
        print(f"    Metadata: {r.get('metadata', {})}")
        print(f"    Images: {len(r.get('images', []))}")


# --- Main ------------------------------------------------------------------


def main():
    demo_multimodal_documents()
    asyncio.run(demo_image_captioner())
    asyncio.run(demo_multimodal_rag())
    asyncio.run(demo_vision_reasoning())
    asyncio.run(demo_agent_with_multimodal_rag())
    asyncio.run(demo_manual_ingestion())

    print("\n" + "=" * 60)
    print("All multimodal RAG demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
