#!/bin/bash
#
# Teacher-Student Distillation Comparison
# =======================================
# 
# Runs teacher-student pairs with live output and finds cases where
# students outperform teachers.
#
# Usage: ./run_comparison.sh [epochs]
# Example: ./run_comparison.sh 2
#

set -e

# Define ALL supported teacher-student pairs for comprehensive comparison
declare -a TEACHERS=(
    "bert-base-uncased"              # BERT: 768×12 = 110M params
    "bert-base-cased"                # BERT: 768×12 = 110M params
    "bert-large-uncased"             # BERT-Large: 1024×24 = 340M params  
    "bert-large-cased"               # BERT-Large: 1024×24 = 340M params
    "distilbert-base-uncased"        # DistilBERT: 768×6 = 66M params
    "distilbert-base-cased"          # DistilBERT: 768×6 = 66M params
    "tinybert"                       # TinyBERT
    "prajjwal1/bert-tiny"            # BERT-tiny: 128×2 = 4M params
    "prajjwal1/bert-mini"            # BERT-mini: 256×4 = 11M params
    "prajjwal1/bert-small"           # BERT-small: 512×4 = 29M params
    "prajjwal1/bert-medium"          # BERT-medium: 512×8 = 41M params
)

declare -a STUDENTS=(
    "prajjwal1/bert-tiny"                                        # BERT-tiny: 128×2 = 4M params
    "huawei-noah/TinyBERT_General_4L_312D"                      # TinyBERT: 312×4 = 14M params
    "prajjwal1/bert-mini"                                        # BERT-mini: 256×4 = 11M params
    "prajjwal1/bert-small"                                       # BERT-small: 512×4 = 29M params
    "prajjwal1/bert-medium"                                      # BERT-medium: 512×8 = 41M params
    "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"      # MiniLM: 384×6 = 22M params
)

# Generate all teacher-student combinations
declare -a PAIRS=()
for teacher in "${TEACHERS[@]}"; do
    for student in "${STUDENTS[@]}"; do
        PAIRS+=("$teacher,$student")
    done
done

EPOCHS=${1:-3}

echo "🔬 COMPREHENSIVE Teacher-Student Distillation Comparison"
echo "========================================================"
echo "Will test ALL ${#PAIRS[@]} teacher-student combinations"
echo "Testing on: Patient 570 (single patient for speed)"
echo "Epochs per phase: $EPOCHS"
echo ""
echo "🎯 COMPRESSION RATIOS TO TEST:"
echo "   📊 7B LLAMA → 4M BERT-tiny = 1,750x compression!"
echo "   📊 340M BERT-Large → 4M BERT-tiny = 85x compression!"
echo "   📊 125M OPT → 4M BERT-tiny = 31x compression!"
echo "   📊 110M BERT → 4M BERT-tiny = 27x compression!"
echo ""
echo "Estimated time: $((${#PAIRS[@]} * $EPOCHS * 3)) - $((${#PAIRS[@]} * $EPOCHS * 6)) minutes"
echo ""

echo "👨‍🏫 Teachers: ${TEACHERS[*]}"
echo "👨‍🎓 Students: ${STUDENTS[*]}"
echo ""

echo "📝 All ${#PAIRS[@]} pairs to test:"
for i in "${!PAIRS[@]}"; do
    IFS=',' read -r teacher student <<< "${PAIRS[$i]}"
    echo "   $((i+1)). $teacher → $student"
done

echo ""
read -p "Continue? [y/N]: " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

# Create results directory
mkdir -p distillation_experiments

for i in "${!PAIRS[@]}"; do
    IFS=',' read -r teacher student <<< "${PAIRS[$i]}"
    
    echo ""
    echo "################################################################################"
    echo "🔄 EXPERIMENT $((i+1))/${#PAIRS[@]}: $teacher → $student"
    echo "################################################################################"
    
    START_TIME=$(date +%s)
    
    # Run with live output on single patient (570) for speed, no checkpoints to save disk space
    if bash scripts/pipelines/distill_pipeline.sh \
        --teacher "$teacher" \
        --student "$student" \
        --patients 570 \
        --dataset ohiot1dm \
        --teacher-epochs "$EPOCHS" \
        --student-epochs "$EPOCHS" \
        --distill-epochs "$EPOCHS" \
        --seed 42 \
        --no-checkpoints; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "✅ Experiment $((i+1)) completed in $((DURATION / 60))m $((DURATION % 60))s"
        
        # Show results immediately
        echo ""
        echo "📊 QUICK RESULTS:"
        
        # Find latest pipeline directory (now in pipeline_runs for single patient)
        LATEST_PIPELINE=$(find distillation_experiments/pipeline_runs -name "pipeline_*" -type d | sort | tail -1)
        
        if [[ -n "$LATEST_PIPELINE" ]]; then
            # Extract metrics
            TEACHER_JSON="$LATEST_PIPELINE/phase_1_teacher/teacher_training_summary.json"
            STUDENT_JSON="$LATEST_PIPELINE/phase_2_student/student_baseline_summary.json"
            DISTILLED_JSON="$LATEST_PIPELINE/phase_3_distillation/distillation_summary.json"
            
            if [[ -f "$TEACHER_JSON" ]]; then
                TEACHER_RMSE=$(python3 -c "import json; print(json.load(open('$TEACHER_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
                echo "   👨‍🏫 Teacher RMSE: $TEACHER_RMSE"
            fi
            
            if [[ -f "$STUDENT_JSON" ]]; then
                STUDENT_RMSE=$(python3 -c "import json; print(json.load(open('$STUDENT_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
                echo "   👨‍🎓 Student RMSE: $STUDENT_RMSE"
                
                # Check if student beat teacher
                if [[ "$TEACHER_RMSE" != "N/A" && "$STUDENT_RMSE" != "N/A" ]]; then
                    COMPARISON=$(python3 -c "
teacher, student = $TEACHER_RMSE, $STUDENT_RMSE
if student < teacher:
    improvement = ((teacher - student) / teacher) * 100
    print(f'🎉 STUDENT BEAT TEACHER by {improvement:.1f}%!')
else:
    degradation = ((student - teacher) / teacher) * 100
    print(f'📉 Student {degradation:.1f}% worse than teacher')
" 2>/dev/null || echo "Could not compare")
                    echo "   $COMPARISON"
                fi
            fi
            
            if [[ -f "$DISTILLED_JSON" ]]; then
                DISTILLED_RMSE=$(python3 -c "import json; print(json.load(open('$DISTILLED_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
                echo "   🧠 Distilled RMSE: $DISTILLED_RMSE"
                
                # Check distillation effectiveness
                if [[ "$STUDENT_RMSE" != "N/A" && "$DISTILLED_RMSE" != "N/A" ]]; then
                    DISTILL_EFFECT=$(python3 -c "
student, distilled = $STUDENT_RMSE, $DISTILLED_RMSE
if distilled < student:
    improvement = ((student - distilled) / student) * 100
    print(f'✨ Distillation improved by {improvement:.1f}%')
else:
    degradation = ((distilled - student) / student) * 100
    print(f'📉 Distillation degraded by {degradation:.1f}%')
" 2>/dev/null || echo "Could not compare")
                    echo "   $DISTILL_EFFECT"
                fi
            fi
        fi
        
    else
        echo "❌ Experiment $((i+1)) failed"
    fi
    
    echo "📈 Progress: $((i+1))/${#PAIRS[@]} completed"
    
    # Short pause between experiments
    if [ $((i+1)) -lt ${#PAIRS[@]} ]; then
        echo "⏸️  Next experiment in 5 seconds..."
        sleep 5
    fi
done

# Generate comprehensive summary
echo ""
echo "🎉 ALL ${#PAIRS[@]} EXPERIMENTS COMPLETED!"
echo "=========================================="

# Create comprehensive results summary
SUMMARY_FILE="distillation_experiments/comprehensive_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "� COMPREHENSIVE DISTILLATION RESULTS" > "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "Total pairs tested: ${#PAIRS[@]}" >> "$SUMMARY_FILE"
echo "Epochs per phase: $EPOCHS" >> "$SUMMARY_FILE"
echo "=================================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Find all pipeline directories and extract metrics
echo "🔍 Analyzing all results..."
STUDENT_WINS=0
DISTILL_WINS=0

for pipeline_dir in distillation_experiments/pipeline_runs/pipeline_*; do
    if [[ -d "$pipeline_dir" ]]; then
        PIPELINE_NAME=$(basename "$pipeline_dir")
        
        TEACHER_JSON="$pipeline_dir/phase_1_teacher/teacher_training_summary.json"
        STUDENT_JSON="$pipeline_dir/phase_2_student/student_baseline_summary.json"
        DISTILLED_JSON="$pipeline_dir/phase_3_distillation/distillation_summary.json"
        
        if [[ -f "$TEACHER_JSON" && -f "$STUDENT_JSON" ]]; then
            TEACHER_RMSE=$(python3 -c "import json; print(json.load(open('$TEACHER_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
            STUDENT_RMSE=$(python3 -c "import json; print(json.load(open('$STUDENT_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
            
            echo "$PIPELINE_NAME:" >> "$SUMMARY_FILE"
            echo "  Teacher RMSE: $TEACHER_RMSE" >> "$SUMMARY_FILE"
            echo "  Student RMSE: $STUDENT_RMSE" >> "$SUMMARY_FILE"
            
            # Check if student beat teacher
            if [[ "$TEACHER_RMSE" != "N/A" && "$STUDENT_RMSE" != "N/A" ]]; then
                STUDENT_BETTER=$(python3 -c "print('YES' if $STUDENT_RMSE < $TEACHER_RMSE else 'NO')" 2>/dev/null || echo "UNKNOWN")
                if [[ "$STUDENT_BETTER" == "YES" ]]; then
                    ((STUDENT_WINS++))
                    IMPROVEMENT=$(python3 -c "print(f'{(($TEACHER_RMSE - $STUDENT_RMSE) / $TEACHER_RMSE) * 100:.1f}%')" 2>/dev/null || echo "N/A")
                    echo "  🎉 STUDENT BEAT TEACHER by $IMPROVEMENT!" >> "$SUMMARY_FILE"
                fi
                echo "  Student better than teacher: $STUDENT_BETTER" >> "$SUMMARY_FILE"
            fi
            
            if [[ -f "$DISTILLED_JSON" ]]; then
                DISTILLED_RMSE=$(python3 -c "import json; print(json.load(open('$DISTILLED_JSON'))['performance_metrics']['rmse'])" 2>/dev/null || echo "N/A")
                echo "  Distilled RMSE: $DISTILLED_RMSE" >> "$SUMMARY_FILE"
                
                if [[ "$STUDENT_RMSE" != "N/A" && "$DISTILLED_RMSE" != "N/A" ]]; then
                    DISTILL_BETTER=$(python3 -c "print('YES' if $DISTILLED_RMSE < $STUDENT_RMSE else 'NO')" 2>/dev/null || echo "UNKNOWN")
                    if [[ "$DISTILL_BETTER" == "YES" ]]; then
                        ((DISTILL_WINS++))
                    fi
                    echo "  Distillation improved student: $DISTILL_BETTER" >> "$SUMMARY_FILE"
                fi
            fi
            echo "" >> "$SUMMARY_FILE"
        fi
    fi
done

echo ""
echo "📊 FINAL SUMMARY:"
echo "   🎯 Total experiments: ${#PAIRS[@]}"
echo "   🏆 Students that beat teachers: $STUDENT_WINS"
echo "   ✨ Successful distillations: $DISTILL_WINS"
echo ""
echo "📁 Detailed results saved to: $SUMMARY_FILE"
echo "📁 Individual results in: distillation_experiments/pipeline_runs/"
echo ""
echo "🎉 Use this data to find the best teacher-student pairs!"