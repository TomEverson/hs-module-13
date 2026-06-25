#!/bin/bash
# Start Ollama, pull the model, then hand off to supervisord.
set -e

MODEL="${OLLAMA_MODEL:-tinyllama}"

mkdir -p /data/artifacts /var/log/supervisor

# Start Ollama in background
ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
until curl -sf http://localhost:11434/ > /dev/null 2>&1; do sleep 2; done
echo "Ollama ready."

# Pull model only if not cached in the volume
if ollama list | grep -q "^${MODEL}"; then
    echo "Model '${MODEL}' already cached."
else
    echo "Pulling '${MODEL}'..."
    ollama pull "${MODEL}"
fi

# Launch all other services via supervisord (Ollama keeps running in background)
exec supervisord -c /etc/supervisor/conf.d/hw13.conf
