#!/usr/bin/env bash
# Run the focused five-seed OhioT1DM BG reference matrix.
#
# Methods included (and only these methods):
#   1. Baseline BERT teacher
#   2. Fair-sampling BERT teacher (T1 source)
#   3. No-KD BERT-tiny student
#   4. Baseline KD
#   5. T1 (distilled from the fair teacher)
#   6. O2-only (baseline teacher + learned output calibration)
#   7. T1+O2
#
# Each seed is fully self-contained under $PIPELINE_DIR/seed_<seed>/ so the
# teacher, student, and distilled models are paired by initialization seed.
# Existing checkpoints and complete per-patient inference trees are skipped.
#
# Usage:
#   DRY_RUN=1 bash scripts/pipelines/run_bg_baseline_t1_o2_multiseed.sh
#   bash scripts/pipelines/run_bg_baseline_t1_o2_multiseed.sh
#
# Optional environment overrides:
#   PIPELINE_DIR=<output directory>
#   SEEDS=831363,809906
#   EPOCHS=10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" && -f "$ROOT_DIR/venv/bin/activate" ]]; then
  source "$ROOT_DIR/venv/bin/activate"
fi
if [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/venv/bin/python"
else
  PYTHON_BIN="python3"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-10}"
PIPELINE_DIR="${PIPELINE_DIR:-distillation_experiments/all_patients_pipeline/bg_baseline_t1_o2_multiseed}"
PATIENTS="540,544,552,559,563,567,570,575,584,588,591,596"
INFERENCE_SUBPATH="per_patient_inference/time_llm_per_patient_inference_ohiot1dm"

if [[ -z "${SEEDS:-}" ]]; then
  SEEDS=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'scripts/utilities'); from seeds import fixed_seeds; print(','.join(map(str, fixed_seeds)))")
fi
IFS=',' read -r -a SEED_ARR <<< "$SEEDS"

if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1; received: $DRY_RUN" >&2
  exit 2
fi
if [[ ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer; received: $EPOCHS" >&2
  exit 2
fi
for seed in "${SEED_ARR[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed: $seed" >&2
    exit 2
  fi
done

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

find_checkpoint() {
  local directory="$1"
  find "$directory" -path '*/logs/logs_*/checkpoints/checkpoint.pth' -type f 2>/dev/null | sort | tail -1 || true
}

find_distilled_checkpoint() {
  local directory="$1"
  find "$directory" -name student_distilled.pth -type f 2>/dev/null | sort | tail -1 || true
}

inference_complete() {
  local inference_dir="$1"
  local expected_patients="${2:-12}"
  local completed
  completed=$(find "$inference_dir" -name inference_results_reformatted.csv -type f 2>/dev/null | wc -l)
  [[ "$completed" -ge "$expected_patients" ]]
}

train_teacher() {
  local seed="$1"
  local output_dir="$2"
  local label="$3"
  local fair_teacher="$4"
  local checkpoint
  checkpoint=$(find_checkpoint "$output_dir")

  if [[ -n "$checkpoint" ]]; then
    echo "⏭️  $label seed $seed training exists: $checkpoint"
    RESOLVED_CHECKPOINT="$checkpoint"
    return
  fi

  local command=(
    "$PYTHON_BIN" distillation/scripts/train_teachers.py
    --model bert
    --all-patients
    --dataset ohiot1dm
    --seed "$seed"
    --lr 0.001
    --batch-size 32
    --epochs "$EPOCHS"
    --output-dir "$output_dir"
    --config-dir "$output_dir/configs"
  )
  if [[ "$fair_teacher" == "1" ]]; then
    command+=(--fair-teacher --fair-teacher-feature gender)
  fi

  echo "▶ $label seed $seed"
  run_command "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    RESOLVED_CHECKPOINT="$output_dir/<teacher checkpoint>"
    return
  fi

  checkpoint=$(find_checkpoint "$output_dir")
  [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for $label seed $seed" >&2; exit 1; }
  RESOLVED_CHECKPOINT="$checkpoint"
}

train_student() {
  local seed="$1"
  local output_dir="$2"
  local checkpoint
  checkpoint=$(find_checkpoint "$output_dir")

  if [[ -n "$checkpoint" ]]; then
    echo "⏭️  No-KD Student seed $seed training exists: $checkpoint"
    RESOLVED_CHECKPOINT="$checkpoint"
    return
  fi

  echo "▶ No-KD Student seed $seed"
  run_command \
    "$PYTHON_BIN" distillation/scripts/train_students.py \
    --model prajjwal1/bert-tiny \
    --all-patients \
    --dataset ohiot1dm \
    --seed "$seed" \
    --lr 0.001 \
    --batch-size 32 \
    --epochs "$EPOCHS" \
    --output-dir "$output_dir" \
    --config-dir "$output_dir/configs"
  if [[ "$DRY_RUN" == "1" ]]; then
    RESOLVED_CHECKPOINT="$output_dir/<student checkpoint>"
    return
  fi

  checkpoint=$(find_checkpoint "$output_dir")
  [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for no-KD Student seed $seed" >&2; exit 1; }
  RESOLVED_CHECKPOINT="$checkpoint"
}

run_inference() {
  local label="$1"
  local checkpoint="$2"
  local model="$3"
  local seed="$4"
  local stage_dir="$5"
  local inference_dir="$stage_dir/$INFERENCE_SUBPATH"

  if [[ "$DRY_RUN" != "1" ]] && inference_complete "$inference_dir"; then
    echo "⏭️  $label seed $seed inference exists: $inference_dir"
    return
  fi

  echo "▶ $label seed $seed per-patient inference"
  run_command \
    "$PYTHON_BIN" scripts/time_llm/config_generator.py \
    --mode per_patient_inference \
    --checkpoint-path "$checkpoint" \
    --llm_models "$model" \
    --patients "$PATIENTS" \
    --seeds "$seed" \
    --dataset ohiot1dm \
    --data_scenario standardized \
    --pred-lengths 9 \
    --torch-dtype float32 \
    --output_dir "$inference_dir"
  run_command \
    "$PYTHON_BIN" scripts/time_llm/run_experiments.py \
    --experiments_dir "$(dirname "$inference_dir")"

  if [[ "$DRY_RUN" != "1" ]] && ! inference_complete "$inference_dir"; then
    echo "Incomplete per-patient inference for $label seed $seed: $inference_dir" >&2
    exit 1
  fi
}

distill_one() {
  local label="$1"
  local teacher_checkpoint="$2"
  local seed="$3"
  local seed_dir="$4"
  local run_name="$5"
  shift 5

  local run_dir="$seed_dir/phase_3_distillation/$run_name"
  local checkpoint
  checkpoint=$(find_distilled_checkpoint "$run_dir")
  if [[ -n "$checkpoint" ]]; then
    echo "⏭️  $label seed $seed training exists: $checkpoint"
  else
    echo "▶ $label seed $seed"
    run_command \
      "$PYTHON_BIN" distillation/scripts/distill_students.py \
      --teacher bert \
      --student prajjwal1/bert-tiny \
      --all-patients \
      --dataset ohiot1dm \
      --seed "$seed" \
      --lr 0.001 \
      --batch-size 32 \
      --alpha 0.5 \
      --beta 0.5 \
      --distill-epochs "$EPOCHS" \
      --teacher-checkpoint-path "$teacher_checkpoint" \
      --output-dir "$seed_dir/phase_3_distillation" \
      --config-output-dir "$seed_dir/phase_3_distillation" \
      --pipeline-dir "$seed_dir" \
      "$@"
    if [[ "$DRY_RUN" == "1" ]]; then
      checkpoint="$run_dir/<student_distilled.pth>"
    else
      checkpoint=$(find_distilled_checkpoint "$run_dir")
      [[ -n "$checkpoint" ]] || { echo "No checkpoint produced for $label seed $seed" >&2; exit 1; }
    fi
  fi

  run_inference "$label" "$checkpoint" "BERT-tiny" "$seed" "$run_dir"
}

echo "========================================================================"
echo "Focused BG five-seed reference matrix"
echo "Pipeline: $PIPELINE_DIR"
echo "Seeds:    $SEEDS"
echo "Epochs:   $EPOCHS"
echo "Dry run:  $DRY_RUN"
echo "========================================================================"

for seed in "${SEED_ARR[@]}"; do
  echo "============================== SEED $seed =============================="
  seed_dir="$PIPELINE_DIR/seed_$seed"
  teacher_dir="$seed_dir/phase_1_teacher"
  fair_teacher_dir="$seed_dir/phase_1_teacher_fair"
  student_dir="$seed_dir/phase_2_student"

  train_teacher "$seed" "$teacher_dir" "Baseline Teacher" 0
  baseline_teacher_checkpoint="$RESOLVED_CHECKPOINT"
  train_teacher "$seed" "$fair_teacher_dir" "Fair Teacher (T1 source)" 1
  fair_teacher_checkpoint="$RESOLVED_CHECKPOINT"
  train_student "$seed" "$student_dir"
  student_checkpoint="$RESOLVED_CHECKPOINT"

  run_inference "Baseline Teacher" "$baseline_teacher_checkpoint" "BERT" "$seed" "$teacher_dir"
  run_inference "Fair Teacher (T1 source)" "$fair_teacher_checkpoint" "BERT" "$seed" "$fair_teacher_dir"
  run_inference "No-KD Student" "$student_checkpoint" "BERT-tiny" "$seed" "$student_dir"

  distill_one "Baseline KD" "$baseline_teacher_checkpoint" "$seed" "$seed_dir" \
    "bert_to_bert-tiny_all_patients_seed${seed}" \
    --dir-suffix "_seed${seed}"
  distill_one "T1" "$fair_teacher_checkpoint" "$seed" "$seed_dir" \
    "bert_to_bert-tiny_all_patients_fair_teacher_seed${seed}" \
    --dir-suffix "_fair_teacher_seed${seed}"
  distill_one "O2-only" "$baseline_teacher_checkpoint" "$seed" "$seed_dir" \
    "bert_to_bert-tiny_all_patients_o2_gender_seed${seed}" \
    --student-calibration-head \
    --student-calibration-feature gender \
    --dir-suffix "_seed${seed}"
  distill_one "T1+O2" "$fair_teacher_checkpoint" "$seed" "$seed_dir" \
    "bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed${seed}" \
    --student-calibration-head \
    --student-calibration-feature gender \
    --dir-suffix "_fair_teacher_seed${seed}"
done

echo "============================= AGGREGATING ==============================="
run_command \
  "$PYTHON_BIN" scripts/fairness/aggregate_bg_baseline_t1_o2_multiseed.py \
  --suite-dir "$PIPELINE_DIR" \
  --seeds "$SEEDS" \
  --calibration-mode patient-holdout \
  --calibration-folds 2

echo "[focused-bg] Done. Results: $PIPELINE_DIR/baseline_t1_o2_multiseed_results.csv"