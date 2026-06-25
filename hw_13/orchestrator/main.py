"""
Orchestrator — public API gateway.

Routes:
  POST /predict      → http://classifier:8000/predict  (result stored in DB)
  POST /get_message  → http://llm:11434/v1/chat/completions  (OpenAI-compatible)
  GET  /predictions  → list stored prediction records

The orchestrator does NOT run any model itself.
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://localhost:8001")
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
DB_PATH = os.getenv("DB_PATH", "/data/predictions.db")

TIMEOUT = httpx.Timeout(120.0)


# ── database ──────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                text      TEXT    NOT NULL,
                label     TEXT    NOT NULL,
                probability REAL  NOT NULL,
                created_at TEXT   NOT NULL
            )
        """)
    log.info(f"Database ready at {DB_PATH}")


def _store_prediction(text: str, label: str, probability: float):
    ts = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (text, label, probability, created_at) VALUES (?,?,?,?)",
            (text, label, probability, ts),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="Orchestrator", version="1.0.0", lifespan=lifespan)


# ── /predict ──────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: PredictRequest):
    """Forward to the classifier service, store result, and return response."""
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
    data = resp.json()
    try:
        _store_prediction(req.text, data["label"], data["probability"])
    except Exception as e:
        log.warning(f"DB write failed (non-fatal): {e}")
    return data


@app.get("/predictions")
def list_predictions(limit: int = 50):
    """Return the most recent stored predictions."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, text, label, probability, created_at FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


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
