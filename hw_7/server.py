import logging
import os
import time
from uuid import UUID, uuid4

import torch
from fastapi import FastAPI
from pydantic import UUID4, BaseModel, Field, StrictStr
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "typeform/distilbert-base-uncased-mnli")
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."
DEVICE = 0 if torch.cuda.is_available() else -1

classifier = None


def load_model():
    global classifier
    logger.info("Loading zero-shot classifier: %s ...", MODEL_NAME)
    start = time.time()
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=DEVICE,
    )
    logger.info("Model loaded in %.1fs on %s", time.time() - start,
                 "GPU" if DEVICE == 0 else "CPU")


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class GetMessageRequest(BaseModel):
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: UUID4 | None = None


class GetMessageResponse(BaseModel):
    new_msg_text: StrictStr
    dialog_id: UUID4


class Prediction(BaseModel):
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float = Field(ge=0.0, le=1.0)


def classify_text(text: str) -> float:
    result = classifier(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    bot_index = result["labels"].index("bot")
    probability = float(result["scores"][bot_index])
    return max(0.0, min(1.0, probability))


app = FastAPI(title="Echo Chat + Classifier")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": "gpu" if DEVICE == 0 else "cpu"}


@app.post("/get_message", response_model=GetMessageResponse)
def get_message(body: GetMessageRequest):
    logger.info("Echo: dialog=%s msg='%s'", body.dialog_id, body.last_msg_text[:50])
    return GetMessageResponse(new_msg_text=body.last_msg_text, dialog_id=body.dialog_id)


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage):
    prob = classify_text(msg.text)
    logger.info("Predict: dialog=%s text='%s' → prob=%.4f",
                msg.dialog_id, msg.text[:50], prob)
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=round(prob, 6),
    )
