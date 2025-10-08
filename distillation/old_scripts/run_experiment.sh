#!/bin/bash
# Run a single experiment from config file

echo "=== Time-LLM Experiment Runner ==="

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file> [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/flexible/config_bert_d1namo_raw_standardized_001_15epochs.gin"
    echo "  $0 configs/flexible/config_bert_d1namo_raw_standardized_001_15epochs.gin --dry-run"
    echo ""
    echo "Available configs:"
    python scripts/flexible_experiment_runner.py --list-configs
    exit 1
fi

CONFIG_FILE=$1
DRY_RUN=${2:-""}

echo "Running experiment: $CONFIG_FILE"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Mode: Dry Run"
    python scripts/flexible_experiment_runner.py --config $CONFIG_FILE --dry-run
else
    echo "Mode: Full Execution"
    python scripts/flexible_experiment_runner.py --config $CONFIG_FILE
fi
