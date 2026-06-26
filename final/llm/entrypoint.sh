#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-smollm2:135m}"

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
until curl -sf http://localhost:11434/ > /dev/null 2>&1; do sleep 2; done
echo "Ollama ready."

if ollama list | grep -q "^${MODEL}"; then
    echo "Model '${MODEL}' already cached."
else
    echo "Pulling '${MODEL}'..."
    ollama pull "${MODEL}"
fi
echo "Model '${MODEL}' ready."

wait $OLLAMA_PID
