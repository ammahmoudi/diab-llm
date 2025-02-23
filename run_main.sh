#!/bin/bash

# Usage: ./run_main.sh --config_path <path_to_config> --log_level <log_level> --remove_checkpoints <True/False>

# Default values for arguments
CONFIG_PATH="./configs/config_time_llm.gin"
LOG_LEVEL="INFO"
REMOVE_CHECKPOINTS="False"  # Default value for the flag

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_path) CONFIG_PATH="$2"; shift ;;
        --log_level) LOG_LEVEL="$2"; shift ;;
        --remove_checkpoints) REMOVE_CHECKPOINTS="$2"; shift ;;  # Added this line
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Execute the Python script with accelerate launch and provided arguments
accelerate launch --num_processes=1 --num_machines=1 --dynamo_backend=no --mixed_precision bf16 \
    main.py \
    --config_path "$CONFIG_PATH" \
    --log_level "$LOG_LEVEL" \
    --remove_checkpoints "$REMOVE_CHECKPOINTS"  # Pass the flag to Python script
