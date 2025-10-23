#!/bin/bash

# Corrected Metrics Pipeline Runner
# This script runs the complete corrected metrics calculation and extraction pipeline

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}" 
    echo -e "${BLUE}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse command line arguments
EXPERIMENT_DIR=""
RUN_REPLACEMENT=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --experiment_dir) EXPERIMENT_DIR="$2"; shift ;;
        --with_replacement) RUN_REPLACEMENT=true ;;
        --help) 
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --experiment_dir DIR    Process specific experiment directory"
            echo "  --with_replacement      Also run true value replacement first"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; echo "Use --help for usage info"; exit 1 ;;
    esac
    shift
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

print_header "CORRECTED METRICS PIPELINE"

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Optional true value replacement
if [ "$RUN_REPLACEMENT" = true ]; then
    print_info "Running true value replacement first..."
    
    if [ -n "$EXPERIMENT_DIR" ]; then
        bash scripts/run_replace_true_values.sh --experiments-root "$EXPERIMENT_DIR" --auto_confirm
    else
        bash scripts/run_replace_true_values.sh --auto_confirm
    fi
    
    if [ $? -eq 0 ]; then
        print_success "True value replacement completed"
    else
        print_error "True value replacement failed"
        exit 1
    fi
    echo
fi

# Step 2: Calculate corrected metrics
print_info "Calculating corrected metrics from raw CSV files..."

if [ -n "$EXPERIMENT_DIR" ]; then
    python "$PROJECT_ROOT/utils/calculate_corrected_metrics.py" --experiment_dir "$EXPERIMENT_DIR"
else
    python "$PROJECT_ROOT/utils/calculate_corrected_metrics.py"
fi

if [ $? -eq 0 ]; then
    print_success "Corrected metrics calculation completed"
else
    print_error "Corrected metrics calculation failed"
    exit 1
fi

echo

# Step 3: Extract corrected metrics to CSV
print_info "Extracting corrected metrics to CSV files..."

python "$PROJECT_ROOT/utils/extract_corrected_metrics_from_logs.py"

if [ $? -eq 0 ]; then
    print_success "Corrected metrics extraction completed"
else
    print_error "Corrected metrics extraction failed"
    exit 1
fi

echo

print_header "CORRECTED METRICS PIPELINE COMPLETE"
print_success "All steps completed successfully!"
print_info "Check the 'corrected_metrics_by_experiment' directory for CSV files"

# Show summary of created files
if [ -d "$PROJECT_ROOT/corrected_metrics_by_experiment" ]; then
    echo
    print_info "Generated CSV files:"
    for file in "$PROJECT_ROOT/corrected_metrics_by_experiment"/*.csv; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            record_count=$(($(wc -l < "$file") - 1))
            echo "  📄 $filename: $record_count records"
        fi
    done
fi

echo