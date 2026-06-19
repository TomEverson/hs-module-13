# MLOps Project — HumanOrBot / You Are Bot

## Overview

This repository contains coursework for an MLOps module (HS Module 13), building a bot detection system for the [You Are Bot](https://youare.bot) platform (also referred to as HumanOrBot).

The project evolves through homework assignments, from a simple echo bot to a full LLM-powered chatbot with a fine-tuned classifier.

**GitHub:** https://github.com/TomEverson/hs-module-13

## Project Structure

```
mlops/
├── docs/                    # Documentation (this folder)
├── hw_3/                    # HW 3: Practice notebook
├── hw_6/                    # HW 6: Echo Bot (baseline bot)
├── hw_7/                    # HW 7: Zero-shot classifier chat
├── hw_8/                    # HW 8: Fine-tuned bot classifier
├── hw_9/                    # HW 9: LLM-powered bot with Docker Compose
├── design_document.md       # Pipe manufacturing defect detection (unrelated ML design doc)
└── customer_responses.md    # Customer interview analysis for pipe plant
```

## Homework Assignments

### HW 6 — Echo Bot (`echobot`)

- **Package:** `echobot` (v0.0.1)
- **Description:** Simple echo bot for HumanOrBot project
- **Stack:** FastAPI + Streamlit
- **Endpoints:**
  - `GET /health` — Health check (`{"status": "ok"}`)
  - `POST /get_message` — Echoes back the received message
  - `POST /predict` — Returns a random `is_bot_probability` (stub)
- **Ports:** API on 6872, Streamlit UI on 8502
- **Exposure:** SSH reverse tunnel to `158.160.135.246` (requires `portforward_key`)

### HW 7 — Zero-shot Classifier Chat

- **Package:** `hw7-classifier-chat` (v0.1.0)
- **Description:** Echo chat with zero-shot bot classifier
- **Model:** `typeform/distilbert-base-uncased-mnli` (zero-shot classification)
- **Labels:** `bot`, `human`
- **Template:** `"This message was written by a {}."`
- **Endpoints:** `/health`, `/get_message`, `/predict`

### HW 8 — Fine-tuned Bot Classifier

- **Package:** `hw8-bot-classifier` (v0.1.0)
- **Description:** Zero-shot evaluation + fine-tuned bot classifier for You Are Bot v2
- **Models:**
  - Zero-shot: `typeform/distilbert-base-uncased-mnli` (fallback)
  - Fine-tuned: Trained on Colab → `model_checkpoint/best/`
- **Training data:** `data/train.json`, `data/test.json`
- **Strategy:** Fine-tuned binary classifier preferred; zero-shot as fallback
- **Endpoints:** `/health`, `/get_message`, `/predict`

### HW 9 — LLM Bot (Docker Compose)

- **Description:** Full-stack chat bot with LLM + Classifier
- **Stack:** Docker Compose with 3 services:
  1. **llm-server** — llama.cpp running `qwen2.5-0.5b-instruct` (GGUF Q4_K_M)
  2. **bot-backend** — FastAPI with classifier + LLM querying
  3. **chat-ui** — Streamlit frontend
- **Classifier:** Uses HW 8's fine-tuned model (mounted as volume)
- **LLM preprompt:** `preprompt.txt` — instructs the bot to pretend to be human
- **Ports:** LLM on 8080, API on 8000, UI on 8501

### HW 3 — Practice

- Contains a Jupyter notebook (`practice_d3_ipynb_.ipynb`)

## Platform: You Are Bot (youare.bot)

The platform runs at https://youare.bot and provides:

| Feature | Description |
|---------|-------------|
| **Chat** | Play "Catch the Bot" — chat for 60s and guess human/bot |
| **Register API** | Register your own bot or classifier |
| **Leaderboard** | View rankings for humans, bots, and classifiers |

### Leaderboard Scoring

| Metric | Formula |
|--------|---------|
| Human Score | `Accuracy × Samples / (1 + Samples)` |
| Bot Score | `(1 - Human Accuracy) × Samples / (1 + Samples)` |

### Leaderboard Tabs

- **Humans vs Bots** — Human accuracy in detecting bots, bot deceptiveness scores
- **Bots vs Classifiers** — Technical metrics (uptime) and ML metrics (accuracy, ROC-AUC, F1, etc.)

### Registration

To register a bot/classifier on youare.bot:

1. Your service must expose `GET /health` and `POST /predict` endpoints
2. The service must be publicly accessible (see Connectivity below)
3. Fill the registration form at https://youare.bot → Register your api
4. Required fields: Name, Email, Telegram, Bot/Classifier name, Type, Endpoint URL

## Connectivity

### Method 1: SSH Reverse Tunnel (from scripts)

```bash
ssh -f -i portforward_key -N -R 0.0.0.0:{random_port}:localhost:6872 forwarduser@158.160.135.246
```

The remote host exposes `http://158.160.135.246:{random_port}` to the public.

### Method 2: Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:6872
```

Creates a temporary URL like `https://{random}-{words}.trycloudflare.com`

### Method 3: ngrok

```bash
ngrok http 6872
```

## Current State (as of 2026-06-19)

- **echobot (hw_6):** API with `/health`, `/get_message` (echo), `/predict` (random) endpoints
- **SSH tunnel:** The `forwarduser` account on `158.160.135.246` is unavailable
- **Cloudflare Tunnel:** Used as alternative — creates `*.trycloudflare.com` URLs
- **youare.bot registration:** Multiple attempts failed. Backend `http://backend:8321/register_bot` returns generic error. Name pattern enforced: `^[A-Za-z]+$` (letters only). Telegram: `^@\\w+$`
- **HW 9:** Docker Compose stack (llama.cpp + FastAPI + Streamlit) ready but not run due to storage limits. Model downloaded (469MB) then removed. Docker/Colima installed then uninstalled.

## Git History

```
eb20edb hw_9: docker-compose bot with llama.cpp + FastAPI + Streamlit
a28ee38 hw_8
41ada31 hw_8: bot classifier with fine-tuned model
cbb992e HW-7
3e93815 Remove leaked SSH private key, add to .gitignore
ec14487 HW-6
```
