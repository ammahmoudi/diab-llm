#!/bin/bash
# Generate a single distillation configuration file

# Auto-activate virtual environment if not already active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        echo "🔧 Activating virtual environment..."
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
fi

echo "=== Time-LLM Configuration Generator ==="

# Check if required parameters are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <dataset> <data_type> <patient_id> <model> [epochs] [learning_rate]"
    echo "       (generates distillation configuration)"
    echo ""
    echo "Examples:"
    echo "  $0 d1namo raw_standardized 001 bert"
    echo "  $0 ohiot1dm noisy_standardized 570 distilbert 25 0.0005"
    echo ""
    echo "Available datasets and models:"
    python distillation/scripts/flexible_config_generator.py --list-data --list-models
    exit 1
fi

DATASET=$1
DATA_TYPE=$2
PATIENT=$3
MODEL=$4
EPOCHS=${5:-20}
LR=${6:-0.001}

echo "Generating config for:"
echo "  Dataset: $DATASET"
echo "  Data Type: $DATA_TYPE"
echo "  Patient: $PATIENT"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"

python distillation/scripts/flexible_config_generator.py \
    --dataset $DATASET \
    --data-type $DATA_TYPE \
    --patient $PATIENT \
    --model $MODEL \
    --epochs $EPOCHS \
    --lr $LR
