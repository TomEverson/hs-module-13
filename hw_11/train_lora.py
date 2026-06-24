"""
HW-11: LoRA fine-tuning for bot detection with MLflow experiment tracking.

Re-uses the same bot/human classification task from HW-8.
Each run logs LoRA config, training params, val metrics, and model artifacts.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

MLFLOW_EXPERIMENT = "bot-classifier-lora"


def load_data(data_dir: Path):
    with open(data_dir / "train.json") as f:
        dialogs = json.load(f)
    labels_df = pd.read_csv(data_dir / "ytrain.csv")

    label_map = {}
    for _, row in labels_df.iterrows():
        key = (row["dialog_id"], row["participant_index"])
        label_map[key] = int(row["is_bot"])

    texts, labels = [], []
    for dialog_id, messages in dialogs.items():
        for msg in messages:
            key = (dialog_id, msg["participant_index"])
            if key in label_map:
                texts.append(msg["text"])
                labels.append(label_map[key])

    return texts, labels


def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../hw_8/data")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    # LoRA hyperparameters
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", default="q_lin,v_lin",
                        help="Comma-separated attention module names to apply LoRA to")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-split", type=float, default=0.2)
    parser.add_argument("--run-notes", default="")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    # Load and split data
    print("Loading data...")
    texts, labels = load_data(data_dir)
    print(f"Total messages: {len(texts)}, bot ratio: {sum(labels)/len(labels):.3f}")

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    split = dataset.train_test_split(test_size=args.eval_split, seed=42)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=args.max_length)

    train_ds = split["train"].map(preprocess, batched=True)
    eval_ds = split["test"].map(preprocess, batched=True)

    # Build LoRA model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # MLflow run
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = args.run_name or f"r{args.lora_rank}_a{args.lora_alpha}_lr{args.lr}"

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("lora_rank", args.lora_rank)
        mlflow.log_param("lora_alpha", args.lora_alpha)
        mlflow.log_param("lora_dropout", args.lora_dropout)
        mlflow.log_param("lora_target_modules", args.lora_target_modules)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("max_length", args.max_length)
        mlflow.log_param("trainable_params", trainable)
        mlflow.log_param("total_params", total)
        mlflow.log_param("trainable_pct", round(100 * trainable / total, 3))
        mlflow.set_tag("notes", args.run_notes if args.run_notes else "no notes")

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_args = TrainingArguments(
                output_dir=tmp_dir,
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
                save_total_limit=1,
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

            print("Training...")
            train_result = trainer.train()
            mlflow.log_metric("train_loss", train_result.training_loss)
            mlflow.log_metric("train_runtime_s", train_result.metrics["train_runtime"])

            final_metrics = trainer.evaluate()
            print("\nFinal metrics:")
            for k, v in final_metrics.items():
                print(f"  {k}: {v:.4f}")
                # strip the eval_ prefix for cleaner MLflow names
                clean_key = k.replace("eval_", "")
                mlflow.log_metric(clean_key, v)

            # Save model artifacts
            model_save_dir = Path(tmp_dir) / "lora_model"
            trainer.save_model(str(model_save_dir))
            tokenizer.save_pretrained(str(model_save_dir))

            # Log model artifacts to MLflow
            mlflow.log_artifacts(str(model_save_dir), artifact_path="lora_model")

        print(f"\nRun '{run_name}' logged to MLflow experiment '{MLFLOW_EXPERIMENT}'")


if __name__ == "__main__":
    main()
