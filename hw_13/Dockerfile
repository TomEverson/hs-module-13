# Combined image for Fly.io — all 4 services in one container.
# Local dev still uses docker-compose.yaml with per-service Dockerfiles.

FROM python:3.11-slim

# System deps: supervisor + curl (for Ollama installer + health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl supervisor procps zstd && \
    rm -rf /var/lib/apt/lists/*

# Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# CPU-only torch (keeps image ~800 MB instead of 4 GB)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Python dependencies for all services
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    transformers>=4.40.0 \
    mlflow>=2.13.0 \
    httpx>=0.27.0 \
    pydantic>=2.0.0

# Service code (flat layout so uvicorn can find them directly)
COPY classifier/main.py  /app/classifier_service.py
COPY orchestrator/main.py /app/orchestrator_service.py

COPY supervisord.conf /etc/supervisor/conf.d/hw13.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Only the orchestrator (public gateway) is exposed externally
EXPOSE 8000

CMD ["/start.sh"]
