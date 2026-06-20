# Morainet AI Wiki

> 一个轻量、可扩展、易嵌入的 **AI Agent Runtime Framework** —— *“AI Agent 的 Spring Framework”*。

用同一套内核驱动任意大模型，搭出能调工具、有记忆、可编排、可观测的 Agent。
内核零厂商依赖，外部能力（模型 / 向量库 / 工具）均通过接口接入。

## 从这里开始

- 📘 [入门教程：30 分钟搭一个 Agent](Getting-Started) —— 从安装到完整 Agent，无需 API key（本地 Ollama）

## 能做什么

- **Tool Calling** —— `@tool` 自动生成 JSON Schema
- **多 Provider** —— OpenAI / Claude / Gemini / Ollama / DeepSeek（内置 Mock 离线可跑）
- **可插拔推理** —— ToolCalling（默认）/ ReAct
- **流式输出** —— `agent.astream()`
- **记忆** —— 短期窗口 / 长期向量检索（RAG）/ 自动摘要
- **多 Agent** —— A2A 原生协议 · 辩论/评审/分层委托/共享记忆池 · 动态生成 · 资源隔离 · 池化
- **Workflow** —— DAG 编排 + 可视化
- **可观测 / 持久化** —— Hook · Tracing · Debugger · Checkpoint
- **生产化** —— 重试 / token 预算 / 危险工具审批
- **扩展** —— Plugin（entry points）· MCP 集成

## 参考方向（examples/）

| 方向 | 示例 |
| --- | --- |
| 工具调用 | `quickstart.py` |
| 知识 / RAG | `rag_doc_qa.py` |
| 编码助手（harness） | `coding_assistant.py` |
| 多 Agent | `multi_agent.py` |
| 多 Agent 高阶编排 | `multiagent_collaboration_demo.py` |
| 本地 / 流式对话 | `live_ollama.py` · `chat.py` |

## 其他文档

- 架构设计：`docs/architecture.md`
- 实现说明与路线：`docs/architecture-v1.3.md`
- 贡献指南：`CONTRIBUTING.md`

> 本 Wiki 收录使用教程；设计文档放在仓库 `docs/` 下。
