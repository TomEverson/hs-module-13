# Homework 6 — Classifier Inference Service Report

## 1. Classifier Contract

### Request: `POST /predict`

| Field              | Type    | Description                       |
|--------------------|---------|-----------------------------------|
| `text`             | string  | Message text to classify          |
| `dialog_id`        | UUID4   | ID of the dialog/conversation     |
| `id`               | UUID4   | ID of this specific message       |
| `participant_index` | int     | Index of the participant (0 or 1) |

### Response: `Prediction`

| Field               | Type    | Description                                    |
|---------------------|---------|------------------------------------------------|
| `id`                | UUID4   | Unique ID of this prediction                   |
| `message_id`        | UUID4   | The message being classified                   |
| `dialog_id`         | UUID4   | Dialog the message belongs to                  |
| `participant_index` | int     | Participant being evaluated                    |
| `is_bot_probability` | float   | Probability (0–1) that the message is from a bot |

### Health: `GET /health`

Returns `{"status": "ok", "model": "<model-name>"}`.

---

## 2. Service Status

```
Model: unitary/toxic-bert
Base URL: http://127.0.0.1:8000
```

### Health Check

```
{"status": "ok", "model": "unitary/toxic-bert"}
```

---

## 3. Classification Results (unitary/toxic-bert)

### Dialog 1 — Message 1

```json
{
    "text": "Hi, how are you?",
    "is_bot_probability": 0.000729,
    "participant_index": 0
}
```

### Dialog 1 — Message 2

```json
{
    "text": "I am not sure, but I can help you write a structured answer.",
    "is_bot_probability": 0.000539,
    "participant_index": 1
}
```

### Dialog 1 — Message 3

```json
{
    "text": "As an AI language model, I can help you solve this task.",
    "is_bot_probability": 0.000568,
    "participant_index": 1
}
```

### Dialog 2 — Message 1 (same text, different dialog)

```json
{
    "text": "Hi, how are you?",
    "is_bot_probability": 0.000729,
    "participant_index": 0
}
```

---

## 4. Observations

| Metric                | Value                          |
|-----------------------|--------------------------------|
| Model                 | unitary/toxic-bert             |
| Latency (per request) | ~20–50 ms                      |
| Max score (msg1)      | 0.000729                       |
| Max score (msg2)      | 0.000539                       |
| Max score (msg3)      | 0.000568                       |
| Max score (dialog2)   | 0.000729                       |

### Analysis

- **Same text, same dialog_id, same participant_index**: returns the same probability (0.000729 for "Hi, how are you?"). This makes sense — the model is stateless and deterministic for identical input.
- **Different texts in the same dialog**: all scored very low (0.0005–0.0007). This is expected because `unitary/toxic-bert` measures toxicity, not bot-likeness. Neutral messages score near zero on all toxicity labels.
- **"As an AI language model..."**: scored 0.000568 — still very low, because the model is detecting toxicity, not AI-generated text. A model like `roberta-base-openai-detector` would give a much higher score for this text.
- **Dialog switching**: has no effect — the service is stateless and does not retain dialog history.

### Key Insight

`unitary/toxic-bert` is the wrong model for bot detection. It measures toxicity probability. For bot detection, use a model trained for AI-text detection like `roberta-base-openai-detector`.

---

## 5. Trying a Different Model

To switch models, restart the service with a new `MODEL_NAME`:

```bash
# Kill existing service
pkill -f "uvicorn inference_service"

# Start with a bot-detection model
MODEL_NAME="roberta-base-openai-detector" \
  /Users/tom/Desktop/mlops/hw_6/.venv/bin/python -m uvicorn inference_service:app \
  --host 0.0.0.0 --port 8000 &
```

Then re-run all curl tests. Expected changes:
- Higher `is_bot_probability` for AI-like text ("As an AI language model...")
- Lower probability for casual human text ("Hi, how are you?")
- Latency may change depending on model size

---

## 6. Validation

Bad request (non-UUID inputs):

```
HTTP 422 Unprocessable Entity
```

Pydantic correctly rejects invalid UUIDs with descriptive error messages about which field failed and why.

---

## 7. Summary

| Check                              | Result  |
|------------------------------------|---------|
| /health endpoint                   | OK      |
| /predict with valid input          | OK      |
| Same text → same probability       | OK      |
| Different texts → different scores | OK      |
| Validation (HTTP 422)              | OK      |
| Model switching via MODEL_NAME env | OK      |
