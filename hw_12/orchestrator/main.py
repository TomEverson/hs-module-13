"""
Orchestrator — public API gateway.

Routes:
  POST /predict      → http://classifier:8000/predict
  POST /get_message  → http://llm:11434/v1/chat/completions  (OpenAI-compatible)

The orchestrator does NOT run any model itself.
"""

import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:135m")

TIMEOUT = httpx.Timeout(120.0)

app = FastAPI(title="Orchestrator", version="1.0.0")


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: PredictRequest):
    """Forward to the classifier service and return its response."""
    url = f"{CLASSIFIER_URL}/predict"
    log.info(f"Forwarding /predict → {url}")
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(url, json=req.model_dump())
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Classifier unreachable: {e}")
    return resp.json()


# ── /get_message ──────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str
    system: str = "You are a helpful assistant."


class MessageResponse(BaseModel):
    reply: str
    model: str


@app.post("/get_message", response_model=MessageResponse)
async def get_message(req: MessageRequest):
    """Transform the user message to OpenAI format and forward to the LLM service."""
    url = f"{LLM_URL}/v1/chat/completions"
    log.info(f"Forwarding /get_message → {url}")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.message},
        ],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"LLM unreachable: {e}")

    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    return MessageResponse(reply=reply, model=data.get("model", OLLAMA_MODEL))


# ── health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
