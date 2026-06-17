import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import UUID4, BaseModel, StrictStr
from transformers import pipeline

MODEL_NAME = os.getenv("MODEL_NAME", "unitary/toxic-bert")
MAX_CHARS = int(os.getenv("MAX_CHARS", "1024"))

app = FastAPI()

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    top_k=None,
    function_to_apply="sigmoid",
    device=device,
)


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage):
    clean_text = msg.text.strip()[:MAX_CHARS]
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text must not be empty after trimming")

    raw_scores = classifier(clean_text)
    if raw_scores and isinstance(raw_scores[0], list):
        raw_scores = raw_scores[0]

    max_score = max(item["score"] for item in raw_scores)
    import uuid
    prediction_id = uuid.uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=round(float(max_score), 6),
    )
