#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-smollm2:135m}"

mkdir -p /data/artifacts

echo "=== Starting Ollama ==="
ollama serve &

echo "Waiting for Ollama..."
until curl -sf http://127.0.0.1:11434/ > /dev/null 2>&1; do sleep 2; done
echo "Ollama ready."

if ollama list | grep -q "^${MODEL}"; then
    echo "Model '${MODEL}' already cached."
else
    echo "Pulling '${MODEL}'..."
    ollama pull "${MODEL}"
fi
echo "Model '${MODEL}' ready."

echo "=== Starting MLflow ==="
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:////data/mlflow.db \
    --artifacts-destination /data/artifacts \
    --serve-artifacts \
    --workers 1 &

echo "=== Starting Classifier ==="
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
MODEL_PATH=/app/model \
uvicorn classifier_service:app --host 0.0.0.0 --port 8001 &

echo "Waiting for classifier..."
until curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; do sleep 2; done
echo "Classifier ready."

echo "=== Starting Orchestrator ==="
export CLASSIFIER_URL=http://127.0.0.1:8001
export LLM_URL=http://127.0.0.1:11434
export OLLAMA_MODEL="${MODEL}"
export DB_PATH=/data/predictions.db
export PREPROMPT_PATH=/app/preprompt.txt
exec uvicorn orchestrator_service:app --host 0.0.0.0 --port 8000
