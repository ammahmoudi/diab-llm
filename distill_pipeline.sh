#!/bin/bash
# Complete 3-Step Knowledge Distillation Pipeline
# Step 1: Train teacher model
# Step 2: Train student model (baseline)  
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

# Parse arguments
TEACHER=""
STUDENT=""
DATASET=""
TEACHER_EPOCHS=""
STUDENT_EPOCHS=""
DISTILL_EPOCHS=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --teacher)
            TEACHER="$2"
            shift 2
            ;;
        --student)
            STUDENT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --teacher-epochs)
            TEACHER_EPOCHS="$2"
            shift 2
            ;;
        --student-epochs)
            STUDENT_EPOCHS="$2"
            shift 2
            ;;
        --distill-epochs)
            DISTILL_EPOCHS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --teacher <model> --student <model> --dataset <id> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--dry-run]"
            exit 1
            ;;
    esac
done

# Set defaults
TEACHER=${TEACHER:-bert}
STUDENT=${STUDENT:-tinybert}
DATASET=${DATASET:-570}
TEACHER_EPOCHS=${TEACHER_EPOCHS:-1}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-1}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-1}
DATA_TYPE="raw_standardized"
LR=0.001

# Create organized pipeline directory structure
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PIPELINE_DIR="distillation_experiments/pipeline_runs/pipeline_${TIMESTAMP}"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
CONFIGS_DIR="$PIPELINE_DIR/configs"

# Create directories
mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$PHASE3_DIR" "$CONFIGS_DIR"

# Validate required parameters
if [[ -z "$TEACHER" || -z "$STUDENT" || -z "$DATASET" ]]; then
    echo "❌ Missing required parameters!"
    echo "Usage: $0 --teacher <model> --student <model> --dataset <id> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 --teacher bert --student tinybert --dataset 570 --teacher-epochs 1 --student-epochs 1 --distill-epochs 1"
    echo "  $0 --teacher distilbert --student tinybert --dataset 570 --teacher-epochs 2 --student-epochs 2 --distill-epochs 2"
    echo ""
    exit 1
fi

echo "Starting 3-Step Knowledge Distillation Pipeline:"
echo "  Dataset: $DATASET"
echo "  Teacher Model: $TEACHER"
echo "  Student Model: $STUDENT"
echo "  Teacher Epochs: $TEACHER_EPOCHS"
echo "  Student Epochs: $STUDENT_EPOCHS"
echo "  Distill Epochs: $DISTILL_EPOCHS"
echo "  Learning Rate: $LR"
echo "  Pipeline Directory: $PIPELINE_DIR"

if [ -n "$DRY_RUN" ]; then
    echo "  Mode: Dry Run"
    echo ""
    echo "=========================================="
    echo "DRY RUN: 3-Step Distillation Pipeline"
    echo "=========================================="
    echo "Step 1: Would train $TEACHER teacher model ($TEACHER_EPOCHS epochs)"
    echo "Step 2: Would train $STUDENT student baseline ($STUDENT_EPOCHS epochs)"  
    echo "Step 3: Would distill $STUDENT from $TEACHER ($DISTILL_EPOCHS epochs)"
    echo ""
    echo "Directory structure that would be created:"
    echo "  $PIPELINE_DIR/"
    echo "    ├── phase_1_teacher/"
    echo "    ├── phase_2_student/"
    echo "    ├── phase_3_distillation/"
    echo "    └── configs/"
    echo ""
    echo "Commands that would be executed:"
    echo "  python distillation/scripts/train_teachers.py --model $TEACHER --dataset $DATASET --epochs $TEACHER_EPOCHS --output-dir \"$PHASE1_DIR\" --config-dir \"$CONFIGS_DIR\""
    echo "  python distillation/scripts/flexible_experiment_runner.py --dataset ohiot1dm --data-type $DATA_TYPE --patients $DATASET --models $STUDENT --epochs $STUDENT_EPOCHS --lr $LR --output-dir \"$PHASE2_DIR\" --pipeline-dir \"$PIPELINE_DIR\""
    echo "  python distillation/scripts/distill_students.py --teacher $TEACHER --student $STUDENT --dataset $DATASET --distill-epochs $DISTILL_EPOCHS --teacher-checkpoint-dir \"$PHASE1_DIR\" --student-config-dir \"$PHASE2_DIR\" --output-dir \"$PHASE3_DIR\" --config-output-dir \"$CONFIGS_DIR\" --pipeline-dir \"$PIPELINE_DIR\""
    exit 0
fi

echo ""
echo "=========================================="
echo "🎓 Step 1/3: Training Teacher Model ($TEACHER)"
echo "=========================================="
echo "Training $TEACHER on patient $DATASET for $TEACHER_EPOCHS epochs..."
echo "Output directory: $PHASE1_DIR"
echo "Config directory: $CONFIGS_DIR"
python distillation/scripts/train_teachers.py --model $TEACHER --dataset $DATASET --epochs $TEACHER_EPOCHS --output-dir "$PHASE1_DIR" --config-dir "$CONFIGS_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Teacher training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "👨‍🎓 Step 2/3: Training Student Baseline ($STUDENT)"
echo "=========================================="
echo "Training baseline $STUDENT on patient $DATASET for $STUDENT_EPOCHS epochs..."
echo "Output directory: $PHASE2_DIR"
python distillation/scripts/flexible_experiment_runner.py \
    --dataset ohiot1dm \
    --data-type $DATA_TYPE \
    --patients $DATASET \
    --models $STUDENT \
    --epochs $STUDENT_EPOCHS \
    --lr $LR \
    --output-dir "$PHASE2_DIR" \
    --pipeline-dir "$PIPELINE_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Student baseline training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "🧠 Step 3/3: Knowledge Distillation ($TEACHER → $STUDENT)"
echo "=========================================="
echo "Distilling knowledge from $TEACHER to $STUDENT for patient $DATASET ($DISTILL_EPOCHS epochs)..."
echo "Output directory: $PHASE3_DIR"
echo "Using teacher from: $PHASE1_DIR"
echo "Using student from: $PHASE2_DIR"
python distillation/scripts/distill_students.py \
    --teacher $TEACHER \
    --student $STUDENT \
    --dataset $DATASET \
    --distill-epochs $DISTILL_EPOCHS \
    --teacher-checkpoint-dir "$PHASE1_DIR" \
    --student-config-dir "$PHASE2_DIR" \
    --output-dir "$PHASE3_DIR" \
    --config-output-dir "$CONFIGS_DIR" \
    --pipeline-dir "$PIPELINE_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Knowledge distillation failed!"
    exit 1
fi

echo ""
echo "🎉 SUCCESS: Complete 3-Step Distillation Pipeline Finished!"
echo "============================================="
echo "✅ Teacher trained: $TEACHER (patient $DATASET, $TEACHER_EPOCHS epochs)"
echo "✅ Student baseline: $STUDENT (patient $DATASET, $STUDENT_EPOCHS epochs)"  
echo "✅ Knowledge distilled: $TEACHER → $STUDENT (patient $DATASET, $DISTILL_EPOCHS epochs)"
echo ""
echo "📁 Results saved in: distillation_experiments/"
echo "🔍 Check logs for detailed performance metrics"