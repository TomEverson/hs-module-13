# HW-11: LoRA Fine-Tuning + MLflow Experiment Tracking

Fine-tunes **DistilBERT** for binary text classification (SST-2 sentiment) using **LoRA** (via PEFT), with all experiments tracked in **MLflow**.

---

## Task

- Model: `distilbert-base-uncased`
- Dataset: [SST-2 / GLUE](https://huggingface.co/datasets/nyu-mll/glue) — binary sentiment (positive / negative), 8 000 train samples, 872 eval samples
- Approach: LoRA adapters on attention projections (`q_lin`, `v_lin`)
- Tracking: MLflow logs params, metrics, and adapter weights for each run

---

## How to run

### On Google Colab (recommended)

1. Upload `hw11_lora_mlflow.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **T4 GPU** → Runtime → Change runtime type
3. Run all cells (Ctrl+F9) — ~15 min total
4. Download `mlruns.zip` and `mlflow.db` from the last cells

### View MLflow UI locally

```bash
cd hw_11
uv sync
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns/mlruns \
  --port 5000
```

Open **http://127.0.0.1:5000**

---

## Experiments

All runs use: `distilbert-base-uncased`, batch size 32, 3 epochs, LoRA targets `q_lin,v_lin`.

### exp1 — Baseline (small LoRA)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 4 |
| LoRA alpha | 8 |
| LoRA dropout | 0.1 |
| Learning rate | 2e-5 |
| Trainable params | 0.985% |

| Metric | Value |
|--------|-------|
| Accuracy | 0.8280 |
| F1 | 0.8256 |
| Precision | 0.8534 |
| Recall | 0.7995 |
| Eval loss | 0.5402 |
| Train loss | 0.4696 |

**Notes**: Baseline with minimal adapter size. Lowest F1 of all runs — too few parameters to capture the task well.

---

### exp2 — Larger rank

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.1 |
| Learning rate | 2e-5 |
| Trainable params | 1.308% |

| Metric | Value |
|--------|-------|
| Accuracy | 0.8314 |
| F1 | 0.8304 |
| Precision | 0.8511 |
| Recall | 0.8108 |
| Eval loss | 0.3669 |
| Train loss | 0.4128 |

**Notes**: Rank-16 with `alpha=2*r`. Better F1 and lower loss than baseline — more adapter capacity helps. Same LR as exp1.

---

### exp3 — Higher learning rate ✅ Best run

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Learning rate | 5e-5 |
| Trainable params | 1.093% |

| Metric | Value |
|--------|-------|
| Accuracy | **0.8394** |
| F1 | **0.8423** |
| Precision | 0.8423 |
| Recall | **0.8423** |
| Eval loss | **0.3495** |
| Train loss | **0.3687** |

**Notes**: Medium rank with 2.5× higher LR. Best result overall — higher LR allows faster convergence and better generalisation within 3 epochs.

---

### exp4 — High dropout

| Parameter | Value |
|-----------|-------|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.3 |
| Learning rate | 2e-5 |
| Trainable params | 1.093% |

| Metric | Value |
|--------|-------|
| Accuracy | 0.8245 |
| F1 | 0.8235 |
| Precision | 0.8440 |
| Recall | 0.8041 |
| Eval loss | 0.3758 |
| Train loss | 0.4396 |

**Notes**: High dropout (0.3) vs exp3. Stronger regularisation hurt performance on this small dataset — model underfits with dropout=0.3 at this LR.

---

## Model artifacts

Each run saves the LoRA adapter weights to MLflow artifacts:

```
lora_adapter/
  adapter_config.json      # LoRA config (rank, alpha, target modules, ...)
  adapter_model.safetensors  # adapter weights only (~1-4 MB)
  tokenizer.json
  tokenizer_config.json
  training_args.bin
```

To load a saved adapter:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

base = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = PeftModel.from_pretrained(base, "<path-to-adapter>")
tokenizer = AutoTokenizer.from_pretrained("<path-to-adapter>")
```

---

## Best run

**exp3** (`r=8, alpha=16, lr=5e-5`) is the best run:

- Highest F1 (0.8423) and accuracy (0.8394)
- Lowest eval and train loss
- Perfectly balanced precision and recall (both 0.8423)

The key factor was the higher learning rate (5e-5 vs 2e-5). With only 3 epochs and a small training subset (8 000 samples), the larger LR allows the LoRA adapter to converge more fully. Increasing rank beyond 8 (exp2) added capacity but not enough to overcome the slower LR. High dropout (exp4) over-regularised and reduced recall on this short training run.
