# HW-12: Microservice Architecture with Docker Compose

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │          Docker Compose network           │
                     │                                          │
 User ──► :8000 ──► orchestrator                               │
                     │   │                                      │
                     │   ├──► classifier:8000  /predict         │
                     │   │        └─ DistilBERT (HW-8 model)   │
                     │   │        └─ logs to mlflow:5000        │
                     │   │                                      │
                     │   └──► llm:11434  /v1/chat/completions   │
                     │            └─ Ollama (smollm2:135m)      │
                     │                                          │
                     │      mlflow:5000  (UI + artifact store)  │
                     └──────────────────────────────────────────┘
```

| Service | Internal address | External port |
|---------|-----------------|---------------|
| orchestrator | `http://orchestrator:8000` | **8000** |
| classifier | `http://classifier:8000` | 8001 |
| llm (Ollama) | `http://llm:11434` | 11434 |
| mlflow | `http://mlflow:5000` | 5000 |

> **LLM port note**: Ollama listens on `11434` (its default). The orchestrator
> calls `http://llm:11434/v1/chat/completions` inside the Docker network.

## Prerequisites

- Docker + Docker Compose v2
- *(Optional)* Trained HW-8 model at `../hw_8/model_checkpoint/best/`
  — if absent the classifier falls back to a public HuggingFace model automatically.

## Quick start

```bash
# 1. (Optional) set a different LLM model
cp .env.example .env        # then edit OLLAMA_MODEL if desired

# 2. Build and start all four services
docker compose up --build
```

First run pulls the Ollama model (~90 MB for `smollm2:135m`). Wait until you
see `"Model '...' ready."` in the logs before testing.

## Test the endpoints

### POST /predict — bot probability

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am a human and I want to buy your product."}'
```

Expected response (`probability` is always in **[0, 1]**):
```json
{"label": "human", "probability": 0.12}
```

### POST /get_message — LLM answer

```bash
curl -X POST http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

Expected response:
```json
{"reply": "The capital of France is Paris.", "model": "smollm2:135m"}
```

Optional custom system prompt:
```bash
curl -X POST http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain LoRA in one sentence.", "system": "You are a concise ML tutor."}'
```

### MLflow UI

Open `http://localhost:5000` to view logged classifier experiments and model artifacts.

### Direct classifier access (bypass orchestrator)

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Buy cheap watches now!!!"}'
```

## Stopping

```bash
docker compose down        # stop, keep volumes
docker compose down -v     # stop and delete all data (Ollama cache, MLflow DB)
```

## Notes

- **No secrets required** — the `.env` file only holds the model name.
- **Orchestrator is a pure gateway** — it contains no model code; all inference
  happens inside `classifier` or `llm`.
- **Model swap** — change `OLLAMA_MODEL` in `.env` to use `tinyllama` (~600 MB)
  or `phi3:mini` (~2.2 GB).
