# Examples — 多方向参考 Agent

Morainet 是**通用 Agent Runtime**。下面这些示例用**同一套内核**搭出不同方向的 agent，
证明运行时的通用性，也作为上手模板。

> 全部**离线可跑**（不需 API key）：框架机制（检索 / 工具 / 路由 / 编排 / 验证）做的是**真事**，
> 语言部分用 `MockProvider`；多数示例可用 `MORAINET_OLLAMA_MODEL=qwen2.5:3b` 切到本地真模型。
>
> **成熟度说明**：基础示例为 **demo 级**参考实现，企业级示例为**接近生产**的参考架构。

## 基础示例

| 方向 | 示例 | 复用的内核能力 |
| --- | --- | --- |
| 工具调用 | `quickstart.py` | `@tool` + 工具调用循环 |
| 知识 / RAG | `rag_doc_qa.py` | `LongMemory` 检索 + Agent 自动注入 |
| 编码 / harness | `coding_assistant.py` | 真实工具(读文件/跑测试) + 验证闭环 |
| 客服 / 分诊 | `multi_agent.py`（Router 部分） | `Router` 路由 |
| 研究 / 写作 | `multi_agent.py`（Pipeline 部分） | `Pipeline` 顺序编排 |
| 多 Agent 协作 | `multi_agent.py`（as_tool 部分） | `Agent.as_tool()` 层级 |
| 流程自动化 | `memory_and_workflow.py` | `Workflow` DAG |
| 长期记忆 | `memory_and_workflow.py` | `LongMemory` |
| 本地 / 隐私 | `live_ollama.py` · `chat.py` | `OllamaProvider`（本机、零成本） |
| 流式对话 | `live_ollama_stream.py` · `chat.py` | `agent.astream()` |
| 推理策略 | `react_and_providers.py` | `ReActStrategy` + 多 Provider |
| 调试 / 持久化 | `debug_and_checkpoint.py` | `Debugger` · `Checkpoint` · `resume` |
| 扩展 | `plugins_mcp_retry.py` | Plugin · MCP · 重试 |

## 企业级实战

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 智能客服 RAG | `enterprise_customer_service_rag.py` | LongMemory + Router + ShortMemory + 工具审批 + 工单系统 |
| 代码工程助手 | `enterprise_code_assistant.py` | 真实工具(AST/Lint/测试) + Workflow DAG + Agent 审查 |
| 自动化运维 | `enterprise_ops_agent.py` | 告警处理 + 预案 Runbook + 巡检 + 部署验证 |
| 多部门协同 | `enterprise_cross_dept_pipeline.py` | Pipeline + GroupChat + Debate + Workflow DAG + Checkpoint |

## 运行

```bash
# 离线（默认）
python examples/rag_doc_qa.py
python examples/coding_assistant.py
python examples/multi_agent.py

# 本地真模型（需 ollama serve + ollama pull qwen2.5:3b）
MORAINET_OLLAMA_MODEL=qwen2.5:3b python examples/coding_assistant.py
```

## 多模态

| 场景 | 示例 | 涉及能力 |
| --- | --- | --- |
| 多模态基础 | `multimodal_basic.py` | ContentPart 类型 + Message 构建器 + 多模态工具注册 + Provider 适配器路由 |
| 多模态 RAG | `multimodal_rag.py` | MultimodalDocument + ImageCaptioner + MultimodalRAG + VisionReasoningChain + Agent 集成 |

> macOS + Homebrew Python：命令行需 `export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`
> （编辑器 F5 已在 `.vscode` 配好）。
