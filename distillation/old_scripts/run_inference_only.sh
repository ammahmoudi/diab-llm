#!/bin/bash
# Run Inference Only

echo "Running Inference on All Models..."

# Get project root dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

DATASET=${1:-"584"}

python scripts/run_inference.py --all --dataset $DATASET
