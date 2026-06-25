#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-tinyllama}"

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait until server is accepting requests
echo "Waiting for Ollama to start..."
until curl -sf http://localhost:11434/ > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama is up."

# Pull model only if not already cached
if ollama list | grep -q "^${MODEL}"; then
    echo "Model '${MODEL}' already cached."
else
    echo "Pulling model: ${MODEL} (this may take a few minutes on first run)..."
    ollama pull "${MODEL}"
    echo "Model '${MODEL}' ready."
fi

# Hand off to the server process
wait $OLLAMA_PID
