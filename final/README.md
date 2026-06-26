# TomBot — Final Project

Bot detector + LLM impersonator, deployed as a microservice stack.

## Architecture

```
                  ┌────────────────────────────────────────────┐
                  │            Docker Compose network           │
                  │                                            │
 User ──► :8000 ──► orchestrator                              │
                  │   │                                        │
                  │   ├──► classifier:8000  /predict           │
                  │   │        └─ DistilBERT (HW-8 model)      │
                  │   │        └─ logs to mlflow:5000           │
                  │   │                                        │
                  │   └──► llm:11434  /v1/chat/completions      │
                  │            └─ Ollama (smollm2:135m)         │
                  │                                            │
                  │      mlflow:5000  (UI + artifact store)    │
                  │      ui:8501      (Streamlit chat UI)       │
                  └────────────────────────────────────────────┘
```

| Service | Port (local) | Role |
|---|---|---|
| orchestrator | 8000 | Public API gateway (youare.bot compatible) |
| classifier | 8001 | DistilBERT bot/human detector |
| llm | 11434 | Ollama LLM (smollm2:135m) |
| mlflow | 5000 | Experiment tracking UI |
| ui | 8501 | Streamlit Turing Arena |

On Fly.io all services run in one container via supervisord (no Streamlit).

## Quick start

```bash
# 1. (optional) choose a different LLM model
cp .env.example .env

# 2. build and start everything
docker compose up --build
```

First run pulls the Ollama model (~90 MB for smollm2:135m). Wait for
`"Model 'smollm2:135m' ready."` in the logs before testing.

## Endpoints

### POST /predict — youare.bot schema

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "buy cheap watches now!!!",
    "dialog_id": "00000000-0000-0000-0000-000000000001",
    "id": "00000000-0000-0000-0000-000000000002",
    "participant_index": 0
  }'
```

```json
{
  "id": "...",
  "message_id": "00000000-0000-0000-0000-000000000002",
  "dialog_id": "00000000-0000-0000-0000-000000000001",
  "participant_index": 0,
  "is_bot_probability": 0.9123
}
```

### POST /get_message — dialog-aware LLM response

```bash
curl -X POST http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{
    "dialog_id": "00000000-0000-0000-0000-000000000001",
    "last_msg_text": "hey what are you up to"
  }'
```

```json
{"new_msg_text": "not much lol, just chilling", "dialog_id": "..."}
```

### FastAPI docs

`http://localhost:8000/docs`

### Streamlit UI

`http://localhost:8501` — Turing Arena: chat with the bot, see live bot-probability scores per message.

### MLflow UI

`http://localhost:5000` — classifier startup runs, experiment history.

### Prediction log

```bash
curl http://localhost:8000/predictions
```

## Fly.io deploy

```bash
bash fly-deploy.sh
```

Public URL: `https://tombot.fly.dev`

## Stopping

```bash
docker compose down      # keep volumes
docker compose down -v   # delete all data
```
