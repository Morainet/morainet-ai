# Morainet AI

> 一个轻量、可扩展、易嵌入的 **AI Agent Runtime Framework**。
> *“AI Agent 的 Spring Framework”*

用同一套内核驱动任意大模型，搭出能调工具、有记忆、可编排、可观测的 Agent。
**内核零厂商依赖**，外部能力（模型 / 向量库 / 工具）均通过接口接入。

## 特性

- **Tool Calling** —— `@tool` 自动从类型注解 + docstring 生成 JSON Schema
- **多 Provider** —— OpenAI / Claude / Gemini / Ollama / DeepSeek（内置 `MockProvider` 离线可跑）
- **可插拔推理** —— `ToolCallingStrategy`（默认）/ `ReActStrategy`，可自定义
- **流式输出** —— `agent.astream()`，OpenAI(SSE) / Ollama(NDJSON) 真流式
- **记忆系统** —— 短期窗口 / 长期向量检索 / 自动摘要压缩
- **多 Agent** —— 层级(`as_tool`) / 顺序(`Pipeline`) / 路由(`Router`)
- **Workflow** —— DAG 编排，环检测 + 拓扑分层并行，可导出 Mermaid/DOT
- **可观测 / 持久化** —— Hook · TraceCollector · Debugger · Checkpoint（含断点恢复）
- **生产化** —— 重试 / token 预算 / 危险工具人工审批
- **扩展** —— Plugin（entry points）· MCP 集成
- **类型安全** —— 全程 Pydantic v2 + `mypy --strict`

## 安装

```bash
pip install -e ".[dev]"     # 需 Python 3.11+
```

## 快速开始

```python
from morainet import Agent, tool
from morainet.providers import OpenAIProvider

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。"""
    return f"{city} 今天晴，26°C"

agent = Agent(provider=OpenAIProvider(model="gpt-4o"), tools=[get_weather])
print(agent.run("上海今天适合穿什么？").final_answer)
```

无 API key？换成本地模型即可（`ollama pull qwen2.5:3b`）：

```python
from morainet.providers import OllamaProvider
agent = Agent(provider=OllamaProvider(model="qwen2.5:3b"), tools=[get_weather])
```

## 示例

`examples/` 下有覆盖 RAG、编码助手、多 Agent、工作流、本地对话等方向的可运行示例（离线即可跑）：

```bash
python examples/quickstart.py        # 工具调用
python examples/rag_doc_qa.py        # 知识 / RAG
python examples/coding_assistant.py  # 编码助手（真实工具 + 验证闭环）
python examples/multi_agent.py       # 多 Agent：层级 / 顺序 / 路由
```

完整清单见 [`examples/README.md`](examples/README.md)。

## 配置

环境变量或 `.env`（前缀 `MORAINET_`）：

```bash
MORAINET_OPENAI_API_KEY=sk-xxx
MORAINET_DEFAULT_MODEL=gpt-4o
```

可选依赖：`pip install -e ".[chroma]"`（ChromaDB）、`".[mcp]"`（MCP）、`".[otel]"`（OpenTelemetry）。

## 测试

```bash
pytest                     # 离线单测（不需 key）
pytest -m live             # 真实端点联调（设好凭证；无凭证自动跳过）
```

CI 在 Python 3.11 / 3.12 上跑 `ruff` + `mypy` + `pytest`（覆盖率门禁 80%）。

## 文档

- 架构设计：[`docs/architecture.md`](docs/architecture.md) · 实现说明与路线：[`docs/architecture-v1.3.md`](docs/architecture-v1.3.md)
- 贡献指南：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- **详细教程见 [GitHub Wiki](../../wiki)**

## License

MIT
