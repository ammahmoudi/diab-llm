#!/bin/bash
# Run Inference Only

echo "Running Inference on All Models..."
cd /home/amma/LLM-TIME

DATASET=${1:-"584"}

python scripts/run_inference.py --all --dataset $DATASET
