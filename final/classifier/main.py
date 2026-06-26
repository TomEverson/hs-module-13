"""
TomBot — Classifier service.

Local: loads fine-tuned HW-8 DistilBERT from volume-mounted MODEL_PATH.
Cloud: falls back to distilbert-base-uncased-finetuned-sst-2-english when path absent.

Model is loaded lazily on first /predict request to keep startup time under
the Fly trial plan's 5-min auto-shutdown window.

POST /predict  →  {"label": "bot"|"human", "probability": float}
"""

import logging
import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
FALLBACK_HF_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

_tokenizer = None
_model = None
_model_source = None
_load_lock = threading.Lock()
_loaded = False


def _do_load():
    global _tokenizer, _model, _model_source, _loaded
    import mlflow  # deferred — mlflow + torch are slow to import
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source = MODEL_PATH if os.path.isdir(MODEL_PATH) else FALLBACK_HF_MODEL
    log.info(f"Loading model from: {source}")

    _model_source = source
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForSequenceClassification.from_pretrained(source)
    _model.eval()
    _loaded = True
    log.info("Model ready.")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("tombot-classifier")
        with mlflow.start_run(run_name="startup"):
            mlflow.log_param("model_source", source)
            mlflow.log_param("architecture", "DistilBertForSequenceClassification")
            mlflow.set_tag("env", "fly.io" if source == FALLBACK_HF_MODEL else "local")
        log.info("Startup run logged to MLflow.")
    except Exception as e:
        log.warning(f"MLflow logging skipped: {e}")


def ensure_loaded():
    if _loaded:
        return
    with _load_lock:
        if _loaded:
            return
        _do_load()


app = FastAPI(title="TomBot Classifier", version="1.0.0")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_source": _model_source, "model_loaded": _loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not _loaded:
        try:
            ensure_loaded()
        except Exception as e:
            log.exception("Model load failed")
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    import torch
    inputs = _tokenizer(
        req.text, return_tensors="pt", truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        logits = _model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    bot_prob = float(probs[1])
    label = "bot" if bot_prob >= 0.5 else "human"
    return PredictResponse(label=label, probability=round(bot_prob, 4))
