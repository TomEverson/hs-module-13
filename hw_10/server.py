import os
import random
from uuid import uuid4

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import UUID4, BaseModel, Field, StrictStr
from transformers import pipeline

CLASSIFIER_MODEL_PATH = os.getenv(
    "CLASSIFIER_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "../hw_8/model_checkpoint/best"),
)
DEVICE = 0 if torch.cuda.is_available() else -1

classifier = None
classifier_type = None

# Varied human-ish responses to keep the bot from looking like an echo machine
_RESPONSES = [
    "haha yeah",
    "lol no way",
    "honestly same",
    "wait what??",
    "that's wild tbh",
    "idk man",
    "right?? exactly",
    "omg yes",
    "hm, fair enough",
    "nah i disagree",
    "lmao ok",
    "sure i guess",
    "depends tbh",
    "wait tell me more",
    "interesting...",
    "true true",
    "not gonna lie same",
    "yeah no totally",
]

_dialog_turns: dict[str, int] = {}


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


def load_classifier():
    global classifier, classifier_type
    path = os.path.abspath(CLASSIFIER_MODEL_PATH)
    if os.path.isdir(path):
        logger.info("Loading fine-tuned classifier from {}", path)
        classifier = pipeline(
            "text-classification",
            model=path,
            tokenizer=path,
            device=DEVICE,
        )
        classifier_type = "fine-tuned"
    else:
        logger.warning("Fine-tuned model not found at {}, using zero-shot fallback", path)
        classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=DEVICE,
        )
        classifier_type = "zero-shot"
    logger.info("Classifier ready: type={}", classifier_type)


def classify_text(text: str) -> float:
    if classifier_type == "fine-tuned":
        result = classifier(text, truncation=True, max_length=128)[0]
        score = float(result["score"])
        return score if result["label"] == "LABEL_1" else 1.0 - score
    else:
        result = classifier(
            text,
            candidate_labels=["bot", "human"],
            hypothesis_template="This message was written by a {}.",
        )
        bot_idx = result["labels"].index("bot")
        return max(0.0, min(1.0, float(result["scores"][bot_idx])))


app = FastAPI(title="hw_10: Bot + Classifier")


@app.on_event("startup")
def startup():
    load_classifier()


@app.get("/health")
def health():
    return {"status": "ok", "classifier": classifier_type}


@app.post("/get_message", response_model=GetMessageResponse)
def get_message(body: GetMessageRequest):
    key = str(body.dialog_id)
    turn = _dialog_turns.get(key, 0)
    _dialog_turns[key] = turn + 1

    text = body.last_msg_text.lower().strip()

    # First turn: greeting-style reply
    if turn == 0:
        reply = random.choice(["hey!", "hi there", "yo", "heyyy", "what's up"])
    elif any(w in text for w in ["bot", "ai", "robot", "human", "real"]):
        reply = random.choice(["lol no im human", "why would i be a bot lmao", "im very much human ok"])
    elif "?" in text:
        reply = random.choice(["idk honestly", "good question lol", "hm not sure", "depends?", "why do you ask"])
    elif any(w in text for w in ["bye", "goodbye", "cya", "later"]):
        reply = random.choice(["bye!", "cya", "later!", "take care"])
    else:
        reply = random.choice(_RESPONSES)

    logger.info("get_message dialog={} turn={} → '{}'", body.dialog_id, turn, reply)
    return GetMessageResponse(new_msg_text=reply, dialog_id=body.dialog_id)


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage):
    prob = classify_text(msg.text)
    logger.info("predict dialog={} text='{}' → {:.4f}", msg.dialog_id, msg.text[:40], prob)
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=round(prob, 6),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
