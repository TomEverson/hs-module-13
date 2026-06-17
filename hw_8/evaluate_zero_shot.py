import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from transformers import pipeline
from tqdm import tqdm


MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."
DEVICE = 0 if torch.cuda.is_available() else -1


def load_data(data_dir: Path):
    with open(data_dir / "train.json") as f:
        dialogs = json.load(f)
    labels_df = pd.read_csv(data_dir / "ytrain.csv")
    return dialogs, labels_df


def get_participant_labels(labels_df: pd.DataFrame):
    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["dialog_id"], row["participant_index"])
        label_map[key] = int(row["is_bot"])
    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dialogs, labels_df = load_data(data_dir)
    label_map = get_participant_labels(labels_df)

    print(f"Dialogs: {len(dialogs)}, Participants: {len(label_map)}")
    print(f"Loading model: {args.model} on {'GPU' if DEVICE == 0 else 'CPU'}...")
    start = time.time()
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
        device=DEVICE,
    )
    print(f"Model loaded in {time.time() - start:.1f}s")

    participant_texts = {}
    for dialog_id, messages in dialogs.items():
        for msg in messages:
            key = (dialog_id, msg["participant_index"])
            if key not in participant_texts:
                participant_texts[key] = []
            participant_texts[key].append(msg["text"])

    if args.max_samples:
        keys = list(participant_texts.keys())[:args.max_samples]
        participant_texts = {k: participant_texts[k] for k in keys}

    print(f"Classifying {len(participant_texts)} participants...")

    y_true = []
    y_pred = []
    y_prob = []

    for (dialog_id, participant_idx), texts in tqdm(participant_texts.items(), desc="Classifying"):
        key = (dialog_id, participant_idx)
        if key not in label_map:
            continue

        probs = []
        for text in texts:
            if not text.strip():
                probs.append(0.5)
                continue
            result = classifier(
                text,
                candidate_labels=CANDIDATE_LABELS,
                hypothesis_template=HYPOTHESIS_TEMPLATE,
            )
            bot_idx = result["labels"].index("bot")
            probs.append(result["scores"][bot_idx])

        avg_prob = float(np.mean(probs))

        y_true.append(label_map[key])
        y_prob.append(avg_prob)
        y_pred.append(1 if avg_prob >= 0.5 else 0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    print("\n" + "=" * 60)
    print("ZERO-SHOT CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {len(y_true)}")
    print(f"Bot ratio in data: {y_true.mean():.4f}")
    print(f"Bot ratio predicted: {y_pred.mean():.4f}")
    print()
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    print()
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
