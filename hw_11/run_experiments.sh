#!/usr/bin/env bash
# Run 4 LoRA experiments with different configurations and log each to MLflow.
# Usage: bash run_experiments.sh
# Prerequisite: uv sync (installs dependencies from pyproject.toml)

set -e

DATA_DIR="../hw_8/data"

echo "============================================"
echo "Experiment 1 — Baseline: small LoRA (r=4)"
echo "============================================"
uv run python train_lora.py \
  --data-dir "$DATA_DIR" \
  --lora-rank 4 \
  --lora-alpha 8 \
  --lora-dropout 0.1 \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --run-name "exp1_r4_a8_lr2e-5" \
  --run-notes "Baseline: small rank-4 LoRA, default LR. Minimal parameter overhead."

echo ""
echo "============================================"
echo "Experiment 2 — Larger rank (r=16)"
echo "============================================"
uv run python train_lora.py \
  --data-dir "$DATA_DIR" \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1 \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --run-name "exp2_r16_a32_lr2e-5" \
  --run-notes "Higher rank (r=16) to capture more complex patterns. alpha=2*r rule of thumb."

echo ""
echo "============================================"
echo "Experiment 3 — Higher learning rate (r=8)"
echo "============================================"
uv run python train_lora.py \
  --data-dir "$DATA_DIR" \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --epochs 3 \
  --batch-size 32 \
  --lr 5e-5 \
  --run-name "exp3_r8_a16_lr5e-5" \
  --run-notes "Medium rank with 2.5x higher LR. Tests if faster convergence helps."

echo ""
echo "============================================"
echo "Experiment 4 — High dropout (r=8, drop=0.3)"
echo "============================================"
uv run python train_lora.py \
  --data-dir "$DATA_DIR" \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.3 \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --run-name "exp4_r8_a16_drop0.3" \
  --run-notes "Aggressive dropout (0.3) to test regularization effect vs exp3 baseline."

echo ""
echo "All experiments complete. Launch the UI with: uv run mlflow ui"
