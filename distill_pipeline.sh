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

# Parse command line arguments
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
        --patients)
            PATIENTS="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --kl-weight)
            KL_WEIGHT="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --teacher <model> --student <model> --patients <patient_ids> --dataset <dataset_name> --seed <seed> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--lr <rate>] [--batch-size <size>] [--alpha <weight>] [--beta <weight>] [--kl-weight <weight>] [--temperature <temp>] [--dry-run]"
            exit 1
            ;;
    esac
done

# Set defaults
TEACHER=${TEACHER:-bert}
STUDENT=${STUDENT:-tinybert}
PATIENTS=${PATIENTS:-570}
DATASET_NAME=${DATASET_NAME:-ohiot1dm}
SEED=${SEED:-238822}

# Auto-detect data path based on dataset name
DATA_PATH="./data/${DATASET_NAME}"
TEACHER_EPOCHS=${TEACHER_EPOCHS:-1}
STUDENT_EPOCHS=${STUDENT_EPOCHS:-1}
DISTILL_EPOCHS=${DISTILL_EPOCHS:-1}
LR=${LR:-0.001}
BATCH_SIZE=${BATCH_SIZE:-32}
ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.5}
KL_WEIGHT=${KL_WEIGHT:-0.1}
TEMPERATURE=${TEMPERATURE:-3.0}
DATA_TYPE="raw_standardized"

# Create organized pipeline directory structure
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PIPELINE_DIR="distillation_experiments/pipeline_runs/pipeline_${TIMESTAMP}"
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"

# Create directories (no centralized configs dir)
mkdir -p "$PHASE1_DIR" "$PHASE2_DIR" "$PHASE3_DIR"

# Validate required parameters
if [[ -z "$TEACHER" || -z "$STUDENT" || -z "$PATIENTS" ]]; then
    echo "❌ Missing required parameters!"
    echo "Usage: $0 --teacher <model> --student <model> --patients <patient_ids> --dataset <dataset_name> --data-path <path> --seed <seed> --teacher-epochs <n> --student-epochs <n> --distill-epochs <n> [--dry-run]"
    echo ""
    echo "Examples:"
    echo "  $0 --teacher bert --student tinybert --patients 570 --dataset ohiot1dm --data-path data --seed 238822 --teacher-epochs 1 --student-epochs 1 --distill-epochs 1"
    echo "  $0 --teacher distilbert --student tinybert --patients 584 --dataset d1namo --data-path data --seed 42 --teacher-epochs 2 --student-epochs 2 --distill-epochs 2"
    echo ""
    exit 1
fi

echo "Starting 3-Step Knowledge Distillation Pipeline:"
echo "  Patient IDs: $PATIENTS"
echo "  Dataset Name: $DATASET_NAME"
echo "  Data Path: $DATA_PATH"
echo "  Seed: $SEED"
echo "  Teacher Model: $TEACHER"
echo "  Student Model: $STUDENT"
echo "  Teacher Epochs: $TEACHER_EPOCHS"
echo "  Student Epochs: $STUDENT_EPOCHS"
echo "  Distill Epochs: $DISTILL_EPOCHS"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Distillation Alpha: $ALPHA"
echo "  Distillation Beta: $BETA"
echo "  KL Weight: $KL_WEIGHT"
echo "  Temperature: $TEMPERATURE"
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
    echo "    ├── phase_1_teacher/ (includes teacher config & model directory)"
    echo "    ├── phase_2_student/ (includes student config & model directory)" 
    echo "    └── phase_3_distillation/ (includes distillation config & model directory)"
    echo ""
    echo "Commands that would be executed:"
    echo "  python distillation/scripts/train_teachers.py --model $TEACHER --patients $PATIENTS --dataset $DATASET_NAME --seed $SEED --lr $LR --batch-size $BATCH_SIZE --epochs $TEACHER_EPOCHS --output-dir \"$PHASE1_DIR\" --config-dir \"$PHASE1_DIR\""
    echo "  python distillation/scripts/train_students.py --model $STUDENT --patients $PATIENTS --dataset $DATASET_NAME --seed $SEED --lr $LR --batch-size $BATCH_SIZE --epochs $STUDENT_EPOCHS --output-dir \"$PHASE2_DIR\" --config-dir \"$PHASE2_DIR\""
    echo "  python distillation/scripts/distill_students.py --teacher $TEACHER --student $STUDENT --patients $PATIENTS --dataset $DATASET_NAME --seed $SEED --lr $LR --batch-size $BATCH_SIZE --alpha $ALPHA --beta $BETA --kl-weight $KL_WEIGHT --temperature $TEMPERATURE --distill-epochs $DISTILL_EPOCHS --teacher-checkpoint-dir \"$PHASE1_DIR\" --student-config-dir \"$PHASE2_DIR\" --output-dir \"$PHASE3_DIR\" --config-output-dir \"$PHASE3_DIR\" --pipeline-dir \"$PIPELINE_DIR\""
    exit 0
fi

echo ""
echo "=========================================="
echo "🎓 Step 1/3: Training Teacher Model ($TEACHER)"
echo "=========================================="
echo "Training $TEACHER on patients $PATIENTS for $TEACHER_EPOCHS epochs..."
echo "Output directory: $PHASE1_DIR"
echo "Config will be saved in: $PHASE1_DIR"
python distillation/scripts/train_teachers.py --model $TEACHER --patients $PATIENTS --dataset $DATASET_NAME --seed $SEED --lr $LR --batch-size $BATCH_SIZE --epochs $TEACHER_EPOCHS --output-dir "$PHASE1_DIR" --config-dir "$PHASE1_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Teacher training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "👨‍🎓 Step 2/3: Training Student Baseline ($STUDENT)"
echo "=========================================="
echo "Training baseline $STUDENT on patients $PATIENTS for $STUDENT_EPOCHS epochs..."
echo "Output directory: $PHASE2_DIR"
echo "Config will be saved in: $PHASE2_DIR"
python distillation/scripts/train_students.py \
    --model $STUDENT \
    --patients $PATIENTS \
    --dataset $DATASET_NAME \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --epochs $STUDENT_EPOCHS \
    --output-dir "$PHASE2_DIR" \
    --config-dir "$PHASE2_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Student baseline training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "🧠 Step 3/3: Knowledge Distillation ($TEACHER → $STUDENT)"
echo "=========================================="
echo "Distilling knowledge from $TEACHER to $STUDENT for patients $PATIENTS ($DISTILL_EPOCHS epochs)..."
echo "Output directory: $PHASE3_DIR"
echo "Using teacher from: $PHASE1_DIR"
echo "Using student from: $PHASE2_DIR"
python distillation/scripts/distill_students.py \
    --teacher $TEACHER \
    --student $STUDENT \
    --patients $PATIENTS \
    --dataset $DATASET_NAME \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --alpha $ALPHA \
    --beta $BETA \
    --kl-weight $KL_WEIGHT \
    --temperature $TEMPERATURE \
    --distill-epochs $DISTILL_EPOCHS \
    --teacher-checkpoint-dir "$PHASE1_DIR" \
    --student-config-dir "$PHASE2_DIR" \
    --output-dir "$PHASE3_DIR" \
    --config-output-dir "$PHASE3_DIR" \
    --pipeline-dir "$PIPELINE_DIR"
if [ $? -ne 0 ]; then
    echo "❌ Knowledge distillation failed!"
    exit 1
fi

echo ""
echo "🎉 SUCCESS: Complete 3-Step Distillation Pipeline Finished!"
echo "============================================="
echo "✅ Teacher trained: $TEACHER (patients $PATIENTS, $TEACHER_EPOCHS epochs)"
echo "✅ Student baseline: $STUDENT (patients $PATIENTS, $STUDENT_EPOCHS epochs)"  
echo "✅ Knowledge distilled: $TEACHER → $STUDENT (patients $PATIENTS, $DISTILL_EPOCHS epochs)"
echo ""
echo "📁 Results saved in: distillation_experiments/"
echo "🔍 Check logs for detailed performance metrics"