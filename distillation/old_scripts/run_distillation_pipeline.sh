#!/bin/bash
# Quick Start: Run Complete Distillation Pipeline

echo "==========================================="
echo "Time-LLM Distillation Pipeline - Quick Start" 
echo "==========================================="

# Get project root dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Set default parameters
DATASET=${1:-"584"}
EPOCHS=${2:-"20"}
DRY_RUN=${3:-""}

echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Mode: Dry Run (configs only)"
    python scripts/pipeline_master.py --full --dataset $DATASET --epochs $EPOCHS --dry-run
else
    echo "Mode: Full Execution"
    python scripts/pipeline_master.py --full --dataset $DATASET --epochs $EPOCHS
fi
