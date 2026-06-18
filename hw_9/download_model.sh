#!/usr/bin/env bash
set -euo pipefail

MODEL_REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_DIR="$(cd "$(dirname "$0")" && pwd)/models"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_PATH" ]]; then
    echo "Model already exists: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
    exit 0
fi

echo "Downloading $MODEL_REPO/$MODEL_FILE ..."
HF_URL="https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE"

if command -v wget &>/dev/null; then
    wget -c "$HF_URL" -O "$MODEL_PATH"
elif command -v curl &>/dev/null; then
    curl -C - -L "$HF_URL" -o "$MODEL_PATH"
else
    echo "Need wget or curl to download the model." >&2
    exit 1
fi

echo "Done:"
ls -lh "$MODEL_PATH"
