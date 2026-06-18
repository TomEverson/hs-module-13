import os
import time
from uuid import uuid4

import httpx
import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import UUID4, BaseModel, Field, StrictStr
from transformers import pipeline

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://llm:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-0.5b-instruct")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "/app/model_checkpoint/best")
DEVICE = 0 if torch.cuda.is_available() else -1

classifier = None
classifier_type = None


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
    if os.path.isdir(CLASSIFIER_MODEL_PATH):
        logger.info("Loading fine-tuned classifier from {}", CLASSIFIER_MODEL_PATH)
        classifier = pipeline(
            "text-classification",
            model=CLASSIFIER_MODEL_PATH,
            tokenizer=CLASSIFIER_MODEL_PATH,
            device=DEVICE,
        )
        classifier_type = "fine-tuned"
    else:
        logger.info("Fine-tuned model not found, loading zero-shot fallback")
        classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=DEVICE,
        )
        classifier_type = "zero-shot"
    logger.info("Classifier loaded: type={}, device={}", classifier_type,
                 "GPU" if DEVICE == 0 else "CPU")


def classify_text(text: str) -> float:
    if classifier_type == "fine-tuned":
        result = classifier(text, truncation=True, max_length=128)[0]
        score = float(result["score"])
        if result["label"] == "LABEL_1":
            return score
        else:
            return 1.0 - score
    else:
        result = classifier(
            text,
            candidate_labels=["bot", "human"],
            hypothesis_template="This message was written by a {}.",
        )
        bot_idx = result["labels"].index("bot")
        return max(0.0, min(1.0, float(result["scores"][bot_idx])))


def query_llm(messages: list[dict]) -> str:
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                f"{LLM_BASE_URL}/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 128,
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("LLM call failed: {}", e)
        return ""


app = FastAPI(title="Bot Backend (LLM + Classifier)")

dialog_history: dict[str, list[dict]] = {}


@app.on_event("startup")
def startup():
    load_classifier()
    logger.info("Waiting for LLM server at {} ...", LLM_BASE_URL)
    for _ in range(30):
        try:
            with httpx.Client(timeout=5.0) as c:
                r = c.get(f"{LLM_BASE_URL.replace('/v1', '')}/health")
                if r.status_code == 200:
                    logger.info("LLM server is ready")
                    break
        except Exception:
            time.sleep(2)
    else:
        logger.warning("LLM server not responding, will use echo fallback")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classifier": classifier_type,
        "llm": LLM_MODEL,
    }


@app.post("/get_message", response_model=GetMessageResponse)
async def get_message(body: GetMessageRequest):
    dialog_key = str(body.dialog_id)
    if dialog_key not in dialog_history:
        dialog_history[dialog_key] = []

    with open("preprompt.txt") as f:
        system_prompt = f.read().strip()

    messages = [{"role": "system", "content": system_prompt}]
    for entry in dialog_history[dialog_key]:
        messages.append({"role": "user", "content": entry["user"]})
        if entry.get("bot"):
            messages.append({"role": "assistant", "content": entry["bot"]})
    messages.append({"role": "user", "content": body.last_msg_text})

    response_text = query_llm(messages)

    if not response_text:
        logger.warning("LLM returned empty, echo fallback")
        response_text = body.last_msg_text

    dialog_history[dialog_key].append({
        "user": body.last_msg_text,
        "bot": response_text,
    })

    logger.info("dialog={} user='{}' bot='{}'",
                body.dialog_id, body.last_msg_text[:40], response_text[:40])
    return GetMessageResponse(new_msg_text=response_text, dialog_id=body.dialog_id)


@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage):
    prob = classify_text(msg.text)
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
