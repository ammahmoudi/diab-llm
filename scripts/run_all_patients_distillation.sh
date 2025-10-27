#!/bin/bash
# Run distillation pipeline on ALL PATIENTS COMBINED
# This script uses the existing distillation pipeline with the --all-patients flag

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

echo "🌍 Running Distillation Pipeline on ALL PATIENTS COMBINED"
echo "==========================================================="
echo ""
echo "This will:"
echo "  1. Train teacher model on combined data from all 12 patients (134K samples)"
echo "  2. Train student baseline on combined data from all 12 patients"
echo "  3. Perform knowledge distillation from teacher to student"
echo ""

# Parse command line arguments or use defaults
TEACHER="${TEACHER:-bert}"
STUDENT="${STUDENT:-prajjwal1/bert-tiny}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-5}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-5}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
SEED="${SEED:-42}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ALPHA="${ALPHA:-0.5}"
BETA="${BETA:-0.5}"

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
        --seed)
            SEED="$2"
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
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --teacher <model>        Teacher model (default: bert)"
            echo "  --student <model>        Student model (default: prajjwal1/bert-tiny)"
            echo "  --teacher-epochs <n>     Teacher training epochs (default: 5)"
            echo "  --student-epochs <n>     Student training epochs (default: 5)"
            echo "  --distill-epochs <n>     Distillation epochs (default: 10)"
            echo "  --seed <n>               Random seed (default: 42)"
            echo "  --lr <rate>              Learning rate (default: 0.001)"
            echo "  --batch-size <size>      Batch size (default: 32)"
            echo "  --alpha <weight>         Ground truth loss weight (default: 0.5)"
            echo "  --beta <weight>          Teacher loss weight (default: 0.5)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --teacher bert --student prajjwal1/bert-tiny --teacher-epochs 10 --distill-epochs 15"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Teacher: $TEACHER ($TEACHER_EPOCHS epochs)"
echo "  Student: $STUDENT ($STUDENT_EPOCHS epochs)"
echo "  Distillation: $DISTILL_EPOCHS epochs"
echo "  Seed: $SEED"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Alpha (ground truth weight): $ALPHA"
echo "  Beta (teacher weight): $BETA"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
PIPELINE_DIR="distillation_experiments/all_patients_pipeline/pipeline_${TIMESTAMP}"

# Create pipeline directory
mkdir -p "$PIPELINE_DIR"

echo "Pipeline directory: $PIPELINE_DIR"
echo ""

# Record start time
PIPELINE_START_TIME=$(date +%s)

# ========================================
# PHASE 1: Train Teacher
# ========================================
echo "=================================================="
echo "🎓 Phase 1/3: Training Teacher Model ($TEACHER)"
echo "=================================================="
PHASE1_DIR="$PIPELINE_DIR/phase_1_teacher"
mkdir -p "$PHASE1_DIR"

python distillation/scripts/train_teachers.py \
    --model $TEACHER \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --epochs $TEACHER_EPOCHS \
    --output-dir "$PHASE1_DIR" \
    --config-dir "$PHASE1_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Teacher training failed!"
    exit 1
fi
echo "✅ Teacher training completed"
echo ""

# Run per-patient inference on teacher model
echo "=================================================="
echo "📊 Running Per-Patient Inference on Teacher"
echo "=================================================="
python distillation/scripts/run_per_patient_inference.py \
    --checkpoint-dir "$PHASE1_DIR" \
    --model-name "teacher_$TEACHER" \
    --dataset ohiot1dm \
    --output-dir "$PHASE1_DIR/per_patient_inference"

if [ $? -ne 0 ]; then
    echo "⚠️  Per-patient inference failed for teacher (continuing...)"
else
    echo "✅ Per-patient inference completed for teacher"
fi
echo ""

# ========================================
# PHASE 2: Train Student Baseline
# ========================================
echo "=================================================="
echo "👨‍🎓 Phase 2/3: Training Student Baseline ($STUDENT)"
echo "=================================================="
PHASE2_DIR="$PIPELINE_DIR/phase_2_student"
mkdir -p "$PHASE2_DIR"

python distillation/scripts/train_students.py \
    --model $STUDENT \
    --all-patients \
    --dataset ohiot1dm \
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
echo "✅ Student baseline training completed"
echo ""

# Run per-patient inference on student baseline
echo "=================================================="
echo "📊 Running Per-Patient Inference on Student Baseline"
echo "=================================================="
python distillation/scripts/run_per_patient_inference.py \
    --checkpoint-dir "$PHASE2_DIR" \
    --model-name "student_baseline_$STUDENT" \
    --dataset ohiot1dm \
    --output-dir "$PHASE2_DIR/per_patient_inference"

if [ $? -ne 0 ]; then
    echo "⚠️  Per-patient inference failed for student baseline (continuing...)"
else
    echo "✅ Per-patient inference completed for student baseline"
fi
echo ""

# ========================================
# PHASE 3: Knowledge Distillation
# ========================================
echo "=================================================="
echo "🧠 Phase 3/3: Knowledge Distillation ($TEACHER → $STUDENT)"
echo "=================================================="
PHASE3_DIR="$PIPELINE_DIR/phase_3_distillation"
mkdir -p "$PHASE3_DIR"

python distillation/scripts/distill_students.py \
    --teacher $TEACHER \
    --student $STUDENT \
    --all-patients \
    --dataset ohiot1dm \
    --seed $SEED \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --alpha $ALPHA \
    --beta $BETA \
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
echo "✅ Knowledge distillation completed"
echo ""

# Run per-patient inference on distilled student
echo "=================================================="
echo "📊 Running Per-Patient Inference on Distilled Student"
echo "=================================================="
python distillation/scripts/run_per_patient_inference.py \
    --checkpoint-dir "$PHASE3_DIR" \
    --model-name "distilled_student_$STUDENT" \
    --dataset ohiot1dm \
    --output-dir "$PHASE3_DIR/per_patient_inference"

if [ $? -ne 0 ]; then
    echo "⚠️  Per-patient inference failed for distilled student (continuing...)"
else
    echo "✅ Per-patient inference completed for distilled student"
fi
echo ""

# ========================================
# Calculate runtime and display summary
# ========================================
PIPELINE_END_TIME=$(date +%s)
TOTAL_RUNTIME=$((PIPELINE_END_TIME - PIPELINE_START_TIME))

echo "🎉 SUCCESS: All-Patients Distillation Pipeline Completed!"
echo "==========================================================="
echo "✅ Phase 1: Teacher trained ($TEACHER_EPOCHS epochs)"
echo "✅ Phase 2: Student baseline trained ($STUDENT_EPOCHS epochs)"
echo "✅ Phase 3: Knowledge distilled ($DISTILL_EPOCHS epochs)"
echo ""
echo "📁 Results saved in: $PIPELINE_DIR"
echo "⏱️  Total Runtime: ${TOTAL_RUNTIME}s ($(($TOTAL_RUNTIME / 60))m $(($TOTAL_RUNTIME % 60))s)"
echo ""
echo "Next steps:"
echo "  - Check logs: find $PIPELINE_DIR -name '*.log'"
echo "  - View metrics: find $PIPELINE_DIR -name '*summary.json'"
echo "  - Compare teacher vs student performance"
echo "  - Run fairness analysis on combined model results"
echo ""
echo "🎯 Ready to test on production data!"
