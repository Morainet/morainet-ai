"""Native multimodal support layer for Morainet AI.

Provides:
- **Content parts** — Text, Image, Audio, File as typed content blocks
- **Message builders** — Create multimodal messages fluently
- **Multimodal tools** — image_understand, ocr, chart_parse, speech_to_text
- **Multimodal RAG** — image+text mixed retrieval and joint reasoning
- **Provider adapters** — transparent routing to GPT-4V, Claude Vision, Gemini, Ollama
"""

from morainet.multimodal.content import (
    AudioPart,
    Content,
    ContentPart,
    ContentType,
    FilePart,
    ImageBase64Part,
    ImageUrlPart,
    TextPart,
    content_has_images,
    content_has_text,
    content_to_openai_blocks,
    content_to_str,
    parse_multimodal_content,
    sanitize_content,
    source_to_part,
    split_text_and_images,
)
from morainet.multimodal.provider_adapter import (
    MultimodalAdapter,
    default_adapter,
    get_adapter,
    register_adapter,
)
from morainet.multimodal.rag import (
    ImageCaptioner,
    ImageTextEncoder,
    MultimodalDocument,
    MultimodalRAG,
    MultimodalRetriever,
    VisionReasoningChain,
    SimpleImageCaptioner,
    SimpleImageTextEncoder,
)
from morainet.multimodal.tools import (
    ImageAnalysisResult,
    chart_parse,
    describe_image,
    image_understand,
    ocr,
    speech_to_text,
)

__all__ = [
    # Content types
    "ContentType",
    "ContentPart",
    "TextPart",
    "ImageUrlPart",
    "ImageBase64Part",
    "AudioPart",
    "FilePart",
    "Content",
    # Content utilities
    "content_to_str",
    "content_has_text",
    "content_has_images",
    "split_text_and_images",
    "content_to_openai_blocks",
    "parse_multimodal_content",
    "sanitize_content",
    "source_to_part",
    # Message builders
    # Multimodal tools
    "image_understand",
    "describe_image",
    "ocr",
    "chart_parse",
    "speech_to_text",
    "ImageAnalysisResult",
    # Multimodal RAG
    "MultimodalDocument",
    "ImageCaptioner",
    "ImageTextEncoder",
    "MultimodalRetriever",
    "MultimodalRAG",
    "VisionReasoningChain",
    "SimpleImageCaptioner",
    "SimpleImageTextEncoder",
    # Provider adapters
    "MultimodalAdapter",
    "default_adapter",
    "get_adapter",
    "register_adapter",
]
