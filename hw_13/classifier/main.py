"""
Classifier microservice.

Local (Docker Compose): loads the fine-tuned HW-8 DistilBERT from a
volume-mounted path (MODEL_PATH=/app/model).

Cloud (Fly.io): MODEL_PATH is absent, so it falls back to the public
HuggingFace model distilbert-base-uncased-finetuned-sst-2-english.

POST /predict  →  {"label": "bot"|"human", "probability": float [0,1]}
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
FALLBACK_HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

_tokenizer = None
_model = None
_model_source = None


def load_model():
    global _tokenizer, _model, _model_source

    if os.path.isdir(MODEL_PATH):
        source = MODEL_PATH
        log.info(f"Loading fine-tuned model from {MODEL_PATH}")
    else:
        source = FALLBACK_HF_MODEL
        log.info(f"MODEL_PATH not found — using HuggingFace fallback: {FALLBACK_HF_MODEL}")

    _model_source = source
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForSequenceClassification.from_pretrained(source)
    _model.eval()
    log.info("Model loaded.")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("bot-classifier-service")
        with mlflow.start_run(run_name="classifier-startup"):
            mlflow.log_param("model_source", source)
            mlflow.log_param("architecture", "DistilBertForSequenceClassification")
            mlflow.log_param("num_labels", 2)
            mlflow.set_tag("labels", "0=human, 1=bot")
            mlflow.set_tag("env", "fly.io" if source == FALLBACK_HF_MODEL else "local")
        log.info("Run logged to MLflow.")
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Classifier Service", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_source": _model_source}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    inputs = _tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    # Fine-tuned HW-8 model: label 1 = bot
    # HF SST-2 fallback: POSITIVE(1) maps to "bot" for demo purposes
    bot_prob = float(probs[1])
    label = "bot" if bot_prob >= 0.5 else "human"

    return PredictResponse(label=label, probability=round(bot_prob, 4))
