# HW-13: Microservice Architecture with MLflow, Classifier, LLM, and Orchestrator

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            Docker Compose network        │
                        │                                         │
  User ──► :8000 ──►  orchestrator                               │
                        │   │                                     │
                        │   ├─► classifier:8000  /predict        │
                        │   │       └─ DistilBERT (HW-8 model)   │
                        │   │       └─ logs to mlflow:5000        │
                        │   │                                     │
                        │   └─► llm:11434  /v1/chat/completions  │
                        │           └─ Ollama (tinyllama)         │
                        │                                         │
                        │       mlflow:5000  (UI + artifact store)│
                        └─────────────────────────────────────────┘
```

| Service | Internal address | External port |
|---------|-----------------|---------------|
| orchestrator | `http://orchestrator:8000` | `8000` |
| classifier | `http://classifier:8000` | `8001` |
| llm (Ollama) | `http://llm:11434` | `11434` |
| mlflow | `http://mlflow:5000` | `5000` |

## Prerequisites

- Docker + Docker Compose
- The trained HW-8 model must exist at `../hw_8/model_checkpoint/best/`

## Quick start

```bash
# 1. (Optional) choose a different LLM model
cp .env.example .env

# 2. Build and start all services
docker compose up --build

# First run pulls the Ollama model (~600 MB for tinyllama).
# Wait until you see "Model 'tinyllama' ready." in the logs.
```

## Test the endpoints

### POST /predict  — bot probability from the classifier

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I am a human and I want to buy your product."}'
```

Expected response:
```json
{"label": "human", "probability": 0.12}
```

The `probability` is always in **[0, 1]** — the model's confidence that the message was written by a bot.

### POST /get_message  — LLM answer

```bash
curl -X POST http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

Expected response:
```json
{"reply": "The capital of France is Paris.", "model": "tinyllama"}
```

You can also pass a custom system prompt:
```bash
curl -X POST http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain LoRA in one sentence.", "system": "You are a concise ML tutor."}'
```

### MLflow UI

Open `http://localhost:5000` to see logged classifier experiments and model artifacts.

## Calling internal services directly

The classifier is also reachable from the host at port 8001:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Buy cheap watches now!!!"}'
```

## Stopping

```bash
docker compose down          # stop, keep volumes
docker compose down -v       # stop and delete all data (model cache, MLflow DB)
```

## Notes

- **Secrets**: no API keys or tokens are required. The `.env` file only contains the model name.
- **LLM port**: Ollama listens on `11434` (its default). The orchestrator calls `http://llm:11434/v1/chat/completions`.
- **Artifact store**: MLflow runs with `--serve-artifacts`, so artifacts are served over HTTP and accessible from all containers.
- **Model swap**: to use a smaller/larger LLM set `OLLAMA_MODEL=smollm2:135m` (90 MB) or `OLLAMA_MODEL=phi3:mini` (2.2 GB) in `.env`.
