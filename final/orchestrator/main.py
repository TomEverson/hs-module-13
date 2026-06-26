"""
TomBot Orchestrator — public API gateway.

Implements the youare.bot API contract exactly:
  POST /predict       → classifier /predict   (IncomingMessage → Prediction)
  POST /get_message   → Ollama LLM            (dialog-aware, GetMessageRequest → GetMessageResponse)
  GET  /predictions   → stored prediction log
  GET  /health
"""

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://localhost:8001")
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "smollm2:135m")
DB_PATH = os.getenv("DB_PATH", "/data/predictions.db")
PREPROMPT_PATH = os.getenv("PREPROMPT_PATH", "/app/preprompt.txt")

TIMEOUT = httpx.Timeout(300.0)

# In-memory dialog history keyed by dialog_id
_dialog_history: dict[str, list[dict]] = {}


def _load_preprompt() -> str:
    try:
        return Path(PREPROMPT_PATH).read_text().strip()
    except Exception:
        return "You are a helpful assistant. Be concise and natural."


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id                TEXT PRIMARY KEY,
                message_id        TEXT NOT NULL,
                dialog_id         TEXT NOT NULL,
                participant_index INTEGER NOT NULL,
                text              TEXT NOT NULL,
                is_bot_probability REAL NOT NULL,
                created_at        TEXT NOT NULL
            )
        """)
    log.info(f"DB ready at {DB_PATH}")


def _store(pred_id, message_id, dialog_id, participant_index, text, prob):
    ts = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO predictions
               (id, message_id, dialog_id, participant_index, text, is_bot_probability, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (str(pred_id), str(message_id), str(dialog_id), participant_index, text, prob, ts),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="TomBot", version="1.0.0", lifespan=lifespan)


# ── youare.bot schemas ────────────────────────────────────────────────────────

class IncomingMessage(BaseModel):
    text: str
    dialog_id: UUID
    id: UUID
    participant_index: int


class Prediction(BaseModel):
    id: UUID
    message_id: UUID
    dialog_id: UUID
    participant_index: int
    is_bot_probability: float = Field(ge=0.0, le=1.0)


class GetMessageRequest(BaseModel):
    dialog_id: UUID
    last_msg_text: str
    last_message_id: UUID | None = None


class GetMessageResponse(BaseModel):
    new_msg_text: str
    dialog_id: UUID


# ── /predict ──────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage):
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{CLASSIFIER_URL}/predict", json={"text": msg.text}
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Classifier unreachable: {e}")

    # classifier returns {"label": "bot"|"human", "probability": float}
    prob = round(float(resp.json()["probability"]), 6)
    pred_id = uuid4()

    try:
        _store(pred_id, msg.id, msg.dialog_id, msg.participant_index, msg.text, prob)
    except Exception as e:
        log.warning(f"DB write failed (non-fatal): {e}")

    return Prediction(
        id=pred_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=prob,
    )


# ── /get_message ──────────────────────────────────────────────────────────────

@app.post("/get_message", response_model=GetMessageResponse)
async def get_message(req: GetMessageRequest):
    key = str(req.dialog_id)
    history = _dialog_history.setdefault(key, [])

    system_prompt = _load_preprompt()
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        if turn.get("bot"):
            messages.append({"role": "assistant", "content": turn["bot"]})
    messages.append({"role": "user", "content": req.last_msg_text})

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"LLM unreachable: {e}")

    reply = resp.json()["choices"][0]["message"]["content"].strip()
    history.append({"user": req.last_msg_text, "bot": reply})
    log.info(f"dialog={key[:8]} user='{req.last_msg_text[:40]}' bot='{reply[:40]}'")

    return GetMessageResponse(new_msg_text=reply, dialog_id=req.dialog_id)


# ── /predictions ──────────────────────────────────────────────────────────────

@app.get("/predictions")
def list_predictions(limit: int = 50):
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT id, message_id, dialog_id, participant_index, text,
                      is_bot_probability, created_at
               FROM predictions ORDER BY rowid DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "tombot-orchestrator"}
