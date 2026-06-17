import json
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import EvalPrediction


BASE_MODEL = "distilbert-base-uncased"
OUTPUT_DIR = "model_checkpoint"


def load_data(data_dir: Path):
    with open(data_dir / "train.json") as f:
        dialogs = json.load(f)
    labels_df = pd.read_csv(data_dir / "ytrain.csv")

    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["dialog_id"], row["participant_index"])
        label_map[key] = int(row["is_bot"])

    return dialogs, label_map


def build_message_dataset(dialogs: dict, label_map: dict):
    texts = []
    labels = []
    for dialog_id, messages in dialogs.items():
        for msg in messages:
            key = (dialog_id, msg["participant_index"])
            if key in label_map:
                texts.append(msg["text"])
                labels.append(label_map[key])

    return texts, labels


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    print("Loading data...")
    dialogs, label_map = load_data(data_dir)
    texts, labels = build_message_dataset(dialogs, label_map)

    bot_ratio = sum(labels) / len(labels)
    print(f"Total messages: {len(texts)}, bot ratio: {bot_ratio:.4f}")

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = train_ds.map(preprocess, batched=True)
    eval_ds = eval_ds.map(preprocess, batched=True)

    if args.no_train:
        print("Skipping training (--no-train)")
        return

    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    final_metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    best_dir = str(output_dir / "best")
    print(f"\nSaving best model to {best_dir}")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    print("Done!")


if __name__ == "__main__":
    main()
