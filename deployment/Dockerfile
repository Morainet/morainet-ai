# Morainet AI — Docker 一键运行模板
#
# 用法：
#   docker build -t morainet-agent -f deployment/Dockerfile .
#   docker run -d --name morainet -p 8000:8000 \
#     -e MORAINET_OPENAI_API_KEY=sk-xxx \
#     morainet-agent
#
# 本地 Ollama 模式：
#   docker run -d --name morainet -p 8000:8000 \
#     -e MORAINET_OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     morainet-agent

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Morainet AI Agent"
LABEL org.opencontainers.image.description="Lightweight AI Agent Runtime"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/Morainet/morainet-ai"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Morainet + FastAPI + optional deps
COPY pyproject.toml .
COPY morainet/ ./morainet/
RUN pip install --no-cache-dir -e ".[openai,redis,marketplace]"

# Copy application & deployment files
COPY deployment/fastapi_demo.py ./app.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
