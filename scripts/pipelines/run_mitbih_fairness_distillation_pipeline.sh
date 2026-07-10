#!/bin/bash
# ============================================================================
# MIT-BIH ECG Time-LLM Fairness Distillation Pipeline
# ============================================================================
# This is the ECG-classification analogue of
# scripts/pipelines/run_fairness_distillation_experiments.sh (the BG pipeline).
#
# Runs, end to end, on a fresh checkout / fresh machine:
#   Phase 0: data prep        (metadata/demographics/beat-index CSVs)
#   Phase 1: teacher training (time_llm_ecg_classifier, e.g. BERT)
#   Phase 2: student baseline (time_llm_ecg_classifier, no distillation)
#   Phase 3: distillation variants (distillation_ecg_classifier):
#              - baseline (no fixes)
#              - T1 fair teacher sampling
#              - O2 student calibration
#              - T1 + O2 (best combo in the BG work)
#              - optionally (RUN_ALL_FIXES=1): K1, O1, K3, K4, O3 individually
#   Phase 4: fairness comparison across every trained variant
#
# Every phase is skipped automatically if its checkpoint + predictions already
# exist, so the script is safe to re-run/resume after an interruption.
#
# Usage:
#   cd /home/amma/LLM-TIME
#   bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# Configuration is via environment variables (all optional, shown with
# defaults below). Example overriding a few:
#   TEACHER_MODEL=BERT STUDENT_MODEL=BERT-tiny TEACHER_EPOCHS=15 \
#     bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# To bootstrap a brand-new machine (create venv + install requirements) before
# running the pipeline, set SETUP_ENV=1:
#   SETUP_ENV=1 bash scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh
#
# All output is tee'd to a timestamped log file under the pipeline directory.
# ============================================================================

set -euo pipefail

# ── Resolve project root and cd into it ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Configuration (override via environment variables) ──────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-BERT}"
STUDENT_MODEL="${STUDENT_MODEL:-TinyBERT}"
SEED="${SEED:-42}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-10}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-10}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-10}"
FAIR_FEATURE="${FAIR_FEATURE:-sex}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
INCLUDE_DUPLICATE_202="${INCLUDE_DUPLICATE_202:-0}"   # 1 = include record 202
PIPELINE_DIR="${PIPELINE_DIR:-experiments/mitbih_fairness_pipeline}"
RUN_ALL_FIXES="${RUN_ALL_FIXES:-0}"                    # 1 = also run K1/O1/K3/K4/O3 individually
SETUP_ENV="${SETUP_ENV:-0}"                            # 1 = create venv + pip install first
LOG_LEVEL="${LOG_LEVEL:-INFO}"

DUP_202_FLAG=""
if [[ "$INCLUDE_DUPLICATE_202" == "1" ]]; then
    DUP_202_FLAG="--include-duplicate-202"
fi

mkdir -p "$PIPELINE_DIR"
LOG_FILE="$PIPELINE_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================================"
echo "🔬 MIT-BIH ECG Fairness Distillation Pipeline"
echo "   Started: $(date)"
echo "   Project root: $PROJECT_ROOT"
echo "   Pipeline dir:  $PIPELINE_DIR"
echo "   Log file:      $LOG_FILE"
echo "   Teacher model: $TEACHER_MODEL (epochs=$TEACHER_EPOCHS)"
echo "   Student model: $STUDENT_MODEL (epochs=$STUDENT_EPOCHS, distill_epochs=$DISTILL_EPOCHS)"
echo "   Fair feature:  $FAIR_FEATURE"
echo "   Run all fixes: $RUN_ALL_FIXES"
echo "========================================================================"
echo ""

# ── Optional environment bootstrap for a brand-new machine ──────────────────
if [[ "$SETUP_ENV" == "1" ]]; then
    echo "🧰 SETUP_ENV=1 — bootstrapping Python environment..."
    if [[ ! -d venv ]]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Environment ready: $(python3 --version), venv=$VIRTUAL_ENV"
    echo ""
fi

# ── Activate venv if present (matches scripts/run_main.sh's own logic) ──────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d venv ]]; then
        source venv/bin/activate
    elif [[ -d .venv ]]; then
        source .venv/bin/activate
    fi
fi
echo "✅ venv: ${VIRTUAL_ENV:-<none, using system python>}"
echo ""

# ── Helper: run a single generated config if not already trained ───────────
# Expects the exact experiment folder (the one containing dataset_mitbih/).
# NOTE: every run of main.py creates a fresh timestamped subdirectory
# (logs/logs_<timestamp>/) via utils/logger.py's setup_logging(), so
# checkpoint/prediction paths are NOT fixed — always glob for the latest one.
latest_checkpoint_path() {
    find "$1/dataset_mitbih/logs" -path "*/checkpoints/checkpoint_best.pth" 2>/dev/null | sort | tail -1
}

latest_predictions_path() {
    find "$1/dataset_mitbih/logs" -maxdepth 2 -name "test_predictions.csv" 2>/dev/null | sort | tail -1
}

run_experiment_if_needed() {
    local exp_dir="$1"
    local label="$2"
    local config_path="$exp_dir/dataset_mitbih/config.gin"
    local ckpt
    ckpt=$(latest_checkpoint_path "$exp_dir")
    local preds
    preds=$(latest_predictions_path "$exp_dir")

    if [[ -n "$ckpt" && -n "$preds" ]]; then
        echo "⏭️  Skipping $label — already trained:"
        echo "     checkpoint:  $ckpt"
        echo "     predictions: $preds"
        return 0
    fi

    if [[ ! -f "$config_path" ]]; then
        echo "❌ Config not found for $label: $config_path"
        exit 1
    fi

    echo "▶ Running $label"
    echo "   Config: $config_path"
    ./scripts/run_main.sh --config_path "$config_path" --log_level "$LOG_LEVEL" --remove_checkpoints False

    ckpt=$(latest_checkpoint_path "$exp_dir")
    if [[ -z "$ckpt" ]]; then
        echo "❌ $label finished but no checkpoint was found under $exp_dir/dataset_mitbih/logs"
        exit 1
    fi
    echo "✅ $label complete"
    echo ""
}

# ── Helper: generate configs and return the seq_256 experiment folder ───────
locate_seq256_experiment_dir() {
    local gen_dir="$1"
    local config_path
    config_path=$(find "$gen_dir" -name config.gin | grep "seq_256" | head -1)
    if [[ -z "$config_path" ]]; then
        echo "❌ No seq_256 config generated under $gen_dir" >&2
        exit 1
    fi
    # config_path = <experiment_folder>/dataset_mitbih/config.gin
    dirname "$(dirname "$config_path")"
}

# ── Phase 0: data prep (idempotent) ─────────────────────────────────────────
echo "========================================================================"
echo "▶ Phase 0/4: MIT-BIH data preparation"
echo "========================================================================"
DATA_DIR="data/mit-bih-arrhythmia"
if [[ -f "$DATA_DIR/metadata_records.csv" && -f "$DATA_DIR/demographics_records.csv" && -f "$DATA_DIR/beat_index.csv" ]]; then
    echo "⏭️  Skipping data prep — CSVs already present in $DATA_DIR"
else
    python scripts/mitbih/build_metadata.py $DUP_202_FLAG
    python scripts/mitbih/build_demographics.py $DUP_202_FLAG
    python scripts/mitbih/prepare_beat_dataset.py $DUP_202_FLAG
fi
echo "✅ Phase 0 complete — $(date)"
echo ""

# ── Phase 1: teacher training ────────────────────────────────────────────────
echo "========================================================================"
echo "▶ Phase 1/4: Teacher training ($TEACHER_MODEL)"
echo "========================================================================"
TEACHER_GEN_DIR="$PIPELINE_DIR/teacher_gen"
python scripts/time_llm/config_generator_mitbih.py \
    --mode train_inference --llm_models "$TEACHER_MODEL" --seeds "$SEED" \
    --epochs "$TEACHER_EPOCHS" --torch-dtype "$TORCH_DTYPE" $DUP_202_FLAG \
    --output_dir "$TEACHER_GEN_DIR"
TEACHER_EXP_DIR=$(locate_seq256_experiment_dir "$TEACHER_GEN_DIR")
run_experiment_if_needed "$TEACHER_EXP_DIR" "teacher ($TEACHER_MODEL)"
TEACHER_CKPT=$(latest_checkpoint_path "$TEACHER_EXP_DIR")
TEACHER_PREDICTIONS=$(latest_predictions_path "$TEACHER_EXP_DIR")
echo "   Teacher checkpoint:  $TEACHER_CKPT"
echo "   Teacher predictions: $TEACHER_PREDICTIONS"
echo ""

# ── Phase 2: student baseline (no distillation) ─────────────────────────────
echo "========================================================================"
echo "▶ Phase 2/4: Student baseline training ($STUDENT_MODEL, no distillation)"
echo "========================================================================"
STUDENT_GEN_DIR="$PIPELINE_DIR/student_baseline_gen"
python scripts/time_llm/config_generator_mitbih.py \
    --mode train_inference --llm_models "$STUDENT_MODEL" --seeds "$SEED" \
    --epochs "$STUDENT_EPOCHS" --torch-dtype "$TORCH_DTYPE" $DUP_202_FLAG \
    --output_dir "$STUDENT_GEN_DIR"
STUDENT_EXP_DIR=$(locate_seq256_experiment_dir "$STUDENT_GEN_DIR")
run_experiment_if_needed "$STUDENT_EXP_DIR" "student baseline ($STUDENT_MODEL)"
STUDENT_PREDICTIONS=$(latest_predictions_path "$STUDENT_EXP_DIR")
echo "   Student baseline predictions: $STUDENT_PREDICTIONS"
echo ""

# ── Phase 3: distillation variants ──────────────────────────────────────────
echo "========================================================================"
echo "▶ Phase 3/4: Distillation variants ($TEACHER_MODEL -> $STUDENT_MODEL)"
echo "========================================================================"

declare -A DISTILL_PREDICTIONS

run_distillation_variant() {
    local label="$1"; shift
    local extra_flags=("$@")
    local gen_dir="$PIPELINE_DIR/distill_${label}_gen"

    python scripts/time_llm/config_generator_mitbih_distillation.py \
        --mode train_inference --teacher-model "$TEACHER_MODEL" \
        --student-models "$STUDENT_MODEL" --teacher-checkpoint-path "$TEACHER_CKPT" \
        --seeds "$SEED" --epochs "$DISTILL_EPOCHS" --torch-dtype "$TORCH_DTYPE" \
        $DUP_202_FLAG --output_dir "$gen_dir" "${extra_flags[@]}"

    local exp_dir
    exp_dir=$(locate_seq256_experiment_dir "$gen_dir")
    run_experiment_if_needed "$exp_dir" "distillation: $label"
    DISTILL_PREDICTIONS["$label"]=$(latest_predictions_path "$exp_dir")
}

run_distillation_variant "baseline"
run_distillation_variant "t1" --fair-teacher --fair-teacher-feature "$FAIR_FEATURE"
run_distillation_variant "o2" --student-calibration --student-calibration-feature "$FAIR_FEATURE"
run_distillation_variant "t1_o2" \
    --fair-teacher --fair-teacher-feature "$FAIR_FEATURE" \
    --student-calibration --student-calibration-feature "$FAIR_FEATURE"

if [[ "$RUN_ALL_FIXES" == "1" ]]; then
    echo "🧪 RUN_ALL_FIXES=1 — also running K1, O1, K3, K4, O3 individually"
    run_distillation_variant "k1" \
        --teacher-calibration --teacher-calibration-feature "$FAIR_FEATURE" \
        --teacher-calibration-offsets '{"M": [0,0,0,0,0], "F": [0,0.3,0.3,0.3,0.3]}'
    run_distillation_variant "o1" \
        --fairness-constraint --fairness-constraint-feature "$FAIR_FEATURE"
    run_distillation_variant "k3" \
        --feature-alignment --feature-alignment-feature "$FAIR_FEATURE"
    run_distillation_variant "k4" \
        --kd-replay --kd-replay-feature "$FAIR_FEATURE" --kd-replay-minority-group F
    run_distillation_variant "o3" \
        --adv-erasure --adv-erasure-feature "$FAIR_FEATURE"
    # T2 (multi-teacher) needs a second, group-specialized teacher checkpoint,
    # which this pipeline does not train by default. Skipped unless you supply
    # SECOND_TEACHER_CHECKPOINT_PATH explicitly.
    if [[ -n "${SECOND_TEACHER_CHECKPOINT_PATH:-}" ]]; then
        run_distillation_variant "t2" \
            --multi-teacher --multi-teacher-feature "$FAIR_FEATURE" --multi-teacher-group0 F \
            --second-teacher-checkpoint-path "$SECOND_TEACHER_CHECKPOINT_PATH"
    else
        echo "⏭️  Skipping T2 (multi-teacher) — set SECOND_TEACHER_CHECKPOINT_PATH to enable"
    fi
fi

echo "✅ Phase 3 complete — $(date)"
echo ""

# ── Phase 4: fairness comparison across every trained variant ──────────────
echo "========================================================================"
echo "▶ Phase 4/4: Fairness comparison"
echo "========================================================================"

PRED_CSV_ARG="teacher=$TEACHER_PREDICTIONS,student_baseline=$STUDENT_PREDICTIONS"
for label in "${!DISTILL_PREDICTIONS[@]}"; do
    PRED_CSV_ARG="$PRED_CSV_ARG,distilled_${label}=${DISTILL_PREDICTIONS[$label]}"
done

FAIRNESS_REPORT="$PIPELINE_DIR/fairness_comparison_${FAIR_FEATURE}.json"
python scripts/fairness/run_mitbih_classifier_fairness.py \
    --prediction-csvs "$PRED_CSV_ARG" \
    --group-column "$FAIR_FEATURE" \
    --output-json "$FAIRNESS_REPORT"

echo "✅ Phase 4 complete — $(date)"
echo ""

echo "========================================================================"
echo "🎉 Pipeline complete — $(date)"
echo "   Teacher checkpoint:      $TEACHER_CKPT"
echo "   Student baseline preds:  $STUDENT_PREDICTIONS"
for label in "${!DISTILL_PREDICTIONS[@]}"; do
    echo "   Distilled ($label) preds: ${DISTILL_PREDICTIONS[$label]}"
done
echo "   Fairness comparison:     $FAIRNESS_REPORT"
echo "   Full log:                $LOG_FILE"
echo "========================================================================"
