#!/bin/bash
# Complete 3-Step Knowledge Distillation Pipeline
# Step 1: Train teacher models
# Step 2: Train student models (baseline)  
# Step 3: Knowledge distillation (student learns from teacher)

# Auto-activate virtual environment if not already active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [[ -f "venv/bin/activate" ]]; then
        echo "🔧 Activating virtual environment..."
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found! Please create one or activate manually."
        exit 1
    fi
fi

echo "🧠 Time-LLM Knowledge Distillation Pipeline"
echo "============================================="

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <patient_id> [epochs] [teacher] [student] [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 ohiot1dm 570                           # Default: 1 epoch, bert→tinybert"
    echo "  $0 ohiot1dm 570 5                         # 5 epochs, bert→tinybert"
    echo "  $0 ohiot1dm 570 1 bert distilbert        # 1 epoch, bert→distilbert"
    echo "  $0 d1namo 001 3 distilbert tinybert      # 3 epochs, distilbert→tinybert"
    echo "  $0 ohiot1dm 570 1 bert tinybert --dry-run # Dry run"
    echo ""
    echo "This will run the complete 3-step pipeline:"
    echo "  1. Train teacher model (e.g., BERT)"
    echo "  2. Train student model baseline (e.g., TinyBERT)"
    echo "  3. Knowledge distillation (student learns from teacher)"
    echo ""
    exit 1
fi

DATASET=$1
PATIENT=$2
EPOCHS=${3:-1}
TEACHER=${4:-bert}
STUDENT=${5:-tinybert}
DATA_TYPE="raw_standardized"  # Standard data type
LR=0.001                      # Standard learning rate
DRY_RUN=""

# Check for dry-run flag in any position
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN="--dry-run"
        break
    fi
done

# Convert comma-separated to space-separated for Python
PATIENTS_ARGS=$(echo $PATIENTS | tr ',' ' ')
MODELS_ARGS=$(echo $MODELS | tr ',' ' ')

echo "Generating and running experiments:"
echo "  Dataset: $DATASET"
echo "  Data Type: $DATA_TYPE"
echo "  Patients: $PATIENTS"
echo "  Models: $MODELS"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"

if [ -n "$DRY_RUN" ]; then
    echo "  Mode: Dry Run"
    echo ""
    echo "=========================================="
    echo "DRY RUN: 3-Step Distillation Pipeline"
    echo "=========================================="
    echo "Step 1: Would train teacher models"
    echo "Step 2: Would train baseline student models"  
    echo "Step 3: Would distill students from teachers"
    echo ""
    python distillation/scripts/flexible_experiment_runner.py \
        --dataset $DATASET \
        --data-type $DATA_TYPE \
        --patients $PATIENTS_ARGS \
        --models $MODELS_ARGS \
        --epochs $EPOCHS \
        --lr $LR \
        --dry-run
else
    echo "  Mode: Full Distillation Pipeline"
    echo ""
    echo "=========================================="
    echo "3-STEP DISTILLATION PIPELINE"
    echo "=========================================="
    
    # Parse models to identify teachers and students
    IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
    TEACHERS=""
    STUDENTS=""
    
    for model in "${MODEL_ARRAY[@]}"; do
        if [[ "$model" == "bert" || "$model" == "distilbert" || "$model" == "gpt2" || "$model" == "albert" || "$model" == "opt_125m" || "$model" == "chronos" ]]; then
            TEACHERS="$TEACHERS$model,"
        else
            STUDENTS="$STUDENTS$model,"
        fi
    done
    
    # Remove trailing commas
    TEACHERS=${TEACHERS%,}
    STUDENTS=${STUDENTS%,}
    
    # If no clear teacher/student separation, default behavior
    if [[ -z "$TEACHERS" || -z "$STUDENTS" ]]; then
        echo "No clear teacher/student separation detected. Running regular training..."
        python distillation/scripts/flexible_experiment_runner.py \
            --dataset $DATASET \
            --data-type $DATA_TYPE \
            --patients $PATIENTS_ARGS \
            --models $MODELS_ARGS \
            --epochs $EPOCHS \
            --lr $LR
    else
        echo "Step 1/3: Training teacher models ($TEACHERS)..."
        python distillation/scripts/flexible_experiment_runner.py \
            --dataset $DATASET \
            --data-type $DATA_TYPE \
            --patients $PATIENTS_ARGS \
            --models "$TEACHERS" \
            --epochs $EPOCHS \
            --lr $LR
        
        echo ""
        echo "Step 2/3: Training baseline student models ($STUDENTS)..."
        python distillation/scripts/flexible_experiment_runner.py \
            --dataset $DATASET \
            --data-type $DATA_TYPE \
            --patients $PATIENTS_ARGS \
            --models "$STUDENTS" \
            --epochs $EPOCHS \
            --lr $LR
        
        echo ""
        echo "Step 3/3: Performing knowledge distillation..."
        
        # Run distillation for each teacher-student combination and each patient
        IFS=',' read -ra TEACHER_ARRAY <<< "$TEACHERS"
        IFS=',' read -ra STUDENT_ARRAY <<< "$STUDENTS"
        IFS=',' read -ra PATIENT_ARRAY <<< "$PATIENTS"
        
        for teacher in "${TEACHER_ARRAY[@]}"; do
            for student in "${STUDENT_ARRAY[@]}"; do
                for patient in "${PATIENT_ARRAY[@]}"; do
                    echo "  Distilling $student from $teacher for patient $patient..."
                    python distillation/scripts/distill_students.py \
                        --teacher "$teacher" \
                        --student "$student" \
                        --dataset "$patient"
                done
            done
        done
        
        echo ""
        echo "✅ Complete 3-step distillation pipeline finished!"
        echo "   - Teacher models: $TEACHERS"
        echo "   - Student models: $STUDENTS" 
        echo "   - Distilled models: Available in distillation_experiments/"
    fi
fi
