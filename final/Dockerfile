# Fly.io monolith — all services in one container managed by supervisord.
# Local dev uses docker-compose.yaml with per-service Dockerfiles instead.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl procps zstd \
    && rm -rf /var/lib/apt/lists/*

# Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# CPU-only torch (~800 MB vs 4 GB for CUDA)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# All service dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    transformers>=4.40.0 \
    mlflow>=2.13.0 \
    httpx>=0.27.0 \
    pydantic>=2.0.0

# Pre-bake the HF fallback model so classifier starts instantly on Fly.io
RUN python3 -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
m = 'distilbert-base-uncased-finetuned-sst-2-english'; \
AutoTokenizer.from_pretrained(m); \
AutoModelForSequenceClassification.from_pretrained(m); \
print('Model cached.')"

# Service code (flat layout for uvicorn)
COPY classifier/main.py  /app/classifier_service.py
COPY orchestrator/main.py /app/orchestrator_service.py

# Preprompt for the LLM system message
COPY ui/preprompt.txt /app/preprompt.txt

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000
CMD ["/start.sh"]
