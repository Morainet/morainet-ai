# Changelog

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [1.4.0] - 2026-08-29

当前开发版本。版本号对齐到实际代码量级（多模态 / 分布式 / 工程治理等模块已落地）。

### 新增

- **Multimodal**（`morainet/multimodal/`）
  - 多模态内容模型：`TextPart` / `ImageUrlPart` / `ImageBase64Part` / `AudioPart` / `FilePart`
  - 内容工具：`content_to_str` / `content_has_text` / `content_has_images` / `split_text_and_images` / `content_to_openai_blocks` / `parse_multimodal_content` / `sanitize_content` / `source_to_part`
  - 多模态工具：`image_understand` / `describe_image` / `ocr` / `chart_parse` / `speech_to_text`
  - 多模态 RAG：`MultimodalDocument` / `MultimodalRetriever` / `MultimodalRAG` / `VisionReasoningChain` / `SimpleImageCaptioner` / `SimpleImageTextEncoder`
  - Provider 适配器：`MultimodalAdapter` / `register_adapter` / `get_adapter` / `default_adapter`
- **Distributed**（`morainet/distributed/`）
  - 任务队列：`Task` / `TaskResult` / `TaskBackend` / `RedisBackend` / `RabbitMQBackend` / `TaskProducer` / `TaskConsumer`
  - DAG 分布式调度：`DistributedScheduler` / `DistributedParallelScheduler` / `DistributedProgressScheduler` / `DistributedNodeExecutor` / `TaskEnvelope` / `TaskStatus`
  - 集群与分片：`AgentCluster` / `ClusterMember` / `EdgeNode` / `ConsistentHashRing` / `SessionShard` / `SessionShardRouter` / `cloud_or_edge`
  - 负载均衡：`Endpoint` / `RoundRobinBalancer` / `WeightedRoundRobinBalancer` / `HybridRouter` / `ProviderShard`
  - 分布式 Checkpoint：`ClusterCheckpointStore` / `HeartbeatCheckpointStore` / `DistributeCheckpointHook`
- **调试工具**
  - 本地调试面板：`morainet-debug`（`PanelStore` / `PanelHook` / `get_panel_store`）
  - Mermaid 导出：`export_mermaid_html` / `export_mermaid_svg` / `export_mermaid_png`
  - CLI：`morainet`（`morainet.cli.main`）
- 企业级示例、部署配置（`deployment/`）与 API 文档

## [1.3.0] - 2026-07

> 里程碑版本号，对应以下已落地的功能批次。

### 新增

- **Multi-agent 进阶**：A2A 协议（`A2AChannel` / `A2ABus`）、`TeamOrchestrator`、`GroupChat` / `Debate` / `DebateTeam` / `ReviewTeam` / `HierarchicalTeam` / `SharedMemoryPool`、`AgentFactory` / `AgentPool` / `AgentSandbox` 隔离
- **Memory 进阶**：`HierarchicalMemory`（分层长期记忆）、`TemporalMemory`（时间感知）
- **MCP 增强**：`MCPConnectionPool`、`MCPResourceCache`
- **Plugin 市场**：`PluginMarketplace`、`PluginManifest`、`RiskLevel`
- **Workflow 调度器**：`SerialScheduler` / `ParallelScheduler` / `ProgressScheduler` + `SchedulerRegistry` 可插拔

## [1.2.0] - 2026-06

### 新增

- **RAG 栈**：`RAGPipeline` / `KnowledgeBase` / `DocumentLoader` / `HybridRetriever` / `CrossEncoderReranker` / `LLMReranker`
- **向量库后端**：`QdrantStore` / `PgVectorStore` / `FaissStore` / `MilvusStore` + `create_vector_store` 工厂
- **推理增强**：`EnhancedReActStrategy` / `PlanSolveReflectStrategy` / `ContextCompressor` / `ToolCache`
- **中文厂商 Provider**：`QwenProvider` / `WenxinProvider` / `ZhipuProvider` / `MoonshotProvider` / `MiniMaxProvider` / `SiliconFlowProvider` / `OpenAICompatibleProvider`
- **模型路由**：`ModelRouter` / `OllamaScheduler` / `multi_model_query` / `multi_ollama_query` / `estimate_complexity`
- **流式扩展**：Claude / Gemini 的 SSE 流式

## [1.1.0] - 2026-05

### 新增

- `CompositeMemory`（组合记忆）
- `response_format` 支持
- 多 Agent：`GroupChat` / `Debate` 拓扑
- 重构 API 文档与 i18n README

## [1.0.0] - 2026-04

首个发布版本。

### 新增

- Agent Core（`Agent` / `AgentResult` / `Message` / `Usage`）
- 多 Provider：OpenAI / Claude / Gemini / Ollama / DeepSeek / Mock + `RetryingProvider`
- 推理策略：`ToolCallingStrategy`（默认）/ `ReActStrategy`
- 流式输出（OpenAI SSE / Ollama NDJSON）
- 记忆：`ShortMemory` / `LongMemory` / `SummarizingMemory`
- 工具系统：`@tool` 装饰器 + 自动 JSON Schema + 参数校验
- Workflow DAG（环检测 + 拓扑分层并行 + Mermaid/DOT 导出）
- Prompt 管理（版本化模板 + 防注入渲染）
- 可观测性：`Hook` / `TraceCollector` / `Debugger` / OTel 导出
- 持久化：`Checkpoint`（内存 / 文件 / SQLite）+ `agent.resume()`
- 生产化：重试 / token 预算 / 连续失败中止 / 危险工具审批
- 扩展：Plugin（entry points）+ MCP（`MCPClient` / `stdio_session`）
- GitHub Actions CI（ruff + mypy + pytest）+ `tests/live` 联调脚手架
