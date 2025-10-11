#!/bin/bash
set -e

# Batch Knowledge Distillation for All OhioT1DM Patients
# This script runs knowledge distillation for all available patients

echo "🔬 Starting Batch Knowledge Distillation for All OhioT1DM Patients"
echo "Teacher: BERT → Student: TinyBERT"
echo "============================================================"

# Get project root dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# List of all ohiot1dm patients
PATIENTS=(540 544 552 567 570 575 588 591 596)

# Distillation parameters
TEACHER="bert"
STUDENT="tinybert"

# Results summary file - in distillation_experiments
SUMMARY_FILE="$PROJECT_ROOT/distillation_experiments/batch_distillation_summary.json"

# Initialize summary
echo "{" > $SUMMARY_FILE
echo "  \"batch_run\": {" >> $SUMMARY_FILE
echo "    \"teacher_model\": \"$TEACHER\"," >> $SUMMARY_FILE
echo "    \"student_model\": \"$STUDENT\"," >> $SUMMARY_FILE
echo "    \"start_time\": \"$(date -Iseconds)\"," >> $SUMMARY_FILE
echo "    \"patients\": [" >> $SUMMARY_FILE

# Track success/failure
SUCCESSFUL_PATIENTS=()
FAILED_PATIENTS=()

# Main distillation loop
for i in "${!PATIENTS[@]}"; do
    PATIENT=${PATIENTS[$i]}
    echo "🎯 Processing Patient $PATIENT ($(($i + 1))/${#PATIENTS[@]})"
    
    # Add comma for JSON formatting (except first entry)
    if [ $i -gt 0 ]; then
        echo "      ," >> $SUMMARY_FILE
    fi
    
    # Start JSON entry for this patient
    echo "      {" >> $SUMMARY_FILE
    echo "        \"patient\": \"$PATIENT\"," >> $SUMMARY_FILE
    echo "        \"start_time\": \"$(date -Iseconds)\"," >> $SUMMARY_FILE
    
    # Run distillation
    if python "$PROJECT_ROOT/distillation/scripts/distill_students.py" --teacher $TEACHER --student $STUDENT --dataset $PATIENT; then
        echo "✅ Patient $PATIENT: Distillation completed successfully"
        SUCCESSFUL_PATIENTS+=($PATIENT)
        echo "        \"status\": \"completed\"," >> $SUMMARY_FILE
        echo "        \"end_time\": \"$(date -Iseconds)\"" >> $SUMMARY_FILE
    else
        echo "❌ Patient $PATIENT: Distillation failed"
        FAILED_PATIENTS+=($PATIENT)
        echo "        \"status\": \"failed\"," >> $SUMMARY_FILE
        echo "        \"end_time\": \"$(date -Iseconds)\"," >> $SUMMARY_FILE
        echo "        \"error\": \"Distillation process failed\"" >> $SUMMARY_FILE
    fi
    
    echo "      }" >> $SUMMARY_FILE
done

# Close JSON structure
echo "    ]," >> $SUMMARY_FILE
echo "    \"end_time\": \"$(date -Iseconds)\"," >> $SUMMARY_FILE
echo "    \"summary\": {" >> $SUMMARY_FILE
echo "      \"total_patients\": ${#PATIENTS[@]}," >> $SUMMARY_FILE
echo "      \"successful\": ${#SUCCESSFUL_PATIENTS[@]}," >> $SUMMARY_FILE
echo "      \"failed\": ${#FAILED_PATIENTS[@]}," >> $SUMMARY_FILE
echo "      \"successful_patients\": [$(IFS=,; echo "\"${SUCCESSFUL_PATIENTS[*]// /\",\")")]," >> $SUMMARY_FILE
echo "      \"failed_patients\": [$(IFS=,; echo "\"${FAILED_PATIENTS[*]// /\",\")")]" >> $SUMMARY_FILE
echo "    }" >> $SUMMARY_FILE
echo "  }" >> $SUMMARY_FILE
echo "}" >> $SUMMARY_FILE

echo "============================================================"
echo "🏁 Batch Distillation Complete!"
echo "✅ Successful: ${#SUCCESSFUL_PATIENTS[@]}/${#PATIENTS[@]} patients"
if [ ${#FAILED_PATIENTS[@]} -gt 0 ]; then
    echo "❌ Failed: ${FAILED_PATIENTS[*]}"
fi
echo "📊 Full summary saved: $SUMMARY_FILE"
echo "============================================================"