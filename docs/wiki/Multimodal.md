# 多模态（Multimodal）

> 模块：`morainet/multimodal/` · 版本：v1.4+ · 示例：`examples/multimodal_basic.py`、`examples/multimodal_rag.py`

多模态能力分三层：**内容模型 → 工具 → 多模态 RAG**，全部通过 Provider 适配器注入模型能力，离线可跑。

## 1. 内容模型（Parts）

统一的部件化内容表示，`parse_multimodal_content` 可从任意来源解析：

```python
from morainet import (
    parse_multimodal_content, content_to_str, content_has_images,
    content_to_openai_blocks, split_text_and_images,
)

parts = parse_multimodal_content("这是什么？https://example.com/a.png")
content_has_images(parts)                       # True
blocks = content_to_openai_blocks(parts)        # -> OpenAI 视觉消息格式
text, images = split_text_and_images(parts)     # 文本与图片分离
```

部件类型：`TextPart` / `ImageUrlPart` / `ImageBase64Part` / `AudioPart` / `FilePart`（`ContentType` 枚举）。
`source_to_part(path_or_url)` 可将本地路径或 URL 一键转成图片部件。

## 2. 工具

统一返回 `ImageAnalysisResult`（`content` / `caption` / `categories` / `scene` / `tags` / `error`）：

```python
from morainet import image_understand, ocr, chart_parse, speech_to_text
```

- `image_understand(source, question)` —— 图像理解（默认走 stub，可注入真实视觉模型）
- `describe_image(source)` —— 图像描述
- `ocr(source)` —— 文字识别
- `chart_parse(source)` —— 图表解析
- `speech_to_text(audio_source)` —— 语音转文字

## 3. 多模态 RAG

```python
from morainet import MultimodalRAG, MultimodalDocument, VisionReasoningChain

doc = MultimodalDocument(text="架构图", image_url="https://example.com/arch.png")
rag = MultimodalRAG(encoder=..., retriever=...)
rag.add_document(doc)
result = await rag.query("图中哪个模块负责调度？")
```

- `MultimodalDocument`：文字 + 图片组合文档
- `ImageCaptioner` / `ImageTextEncoder`：抽象（默认 `SimpleImageCaptioner` / `SimpleImageTextEncoder`，可替换为真实视觉模型）
- `MultimodalRetriever`：多模态检索
- `VisionReasoningChain`：看图-推理链路

## 4. Provider 适配器

```python
from morainet import register_adapter, get_adapter, MultimodalAdapter

class MyVisionAdapter(MultimodalAdapter):
    async def chat(self, messages, **kwargs): ...
    # 实现图像理解 / 描述 / OCR 等方法

register_adapter("my_vision", MyVisionAdapter())
adapter = get_adapter("my_vision")     # default_adapter() 获取默认
```

> 设计约束：不配置适配器时工具走内置 stub，保证离线单测与示例可运行。
