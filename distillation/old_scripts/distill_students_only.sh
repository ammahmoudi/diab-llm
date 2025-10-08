#!/bin/bash
# Perform Distillation Only

echo "Performing Knowledge Distillation..."
cd /home/amma/LLM-TIME

DATASET=${1:-"584"}

python scripts/distill_students.py --all --dataset $DATASET
