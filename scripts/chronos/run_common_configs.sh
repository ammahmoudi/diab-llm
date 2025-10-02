#!/bin/bash

# ==============================================================================
# Auto-Run Config Generation Script
# ==============================================================================
# This script automatically generates common config combinations.
# Edit the variables below, then run the script.
# ==============================================================================

# Configuration variables - EDIT THESE
PATIENTS="570"
MODEL="amazon/chronos-t5-tiny"  
SEED="831363"
CONFIG_GEN="python3 config_generator_chronos.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
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

# Main execution
print_header "GENERATING COMMON CONFIG COMBINATIONS"
echo "Using: Patients=$PATIENTS, Model=$MODEL, Seed=$SEED"
echo

# 1. Basic training configs
print_info "Generating basic training configs..."
$CONFIG_GEN --mode train --patients $PATIENTS --models $MODEL --seeds $SEED
print_success "Basic training config generated!"
echo

# 2. Basic inference configs  
print_info "Generating basic inference configs..."
$CONFIG_GEN --mode inference --patients $PATIENTS --models $MODEL --seeds $SEED
print_success "Basic inference config generated!"
echo

# 3. Noisy data training
print_info "Generating noisy data training configs..."
$CONFIG_GEN --mode train --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED
print_success "Noisy training config generated!"
echo

# 4. Cross-scenario robustness: train clean, test on missing data
print_info "Generating cross-scenario configs (clean→missing)..."
$CONFIG_GEN --mode trained_inference --data_scenario missing_periodic --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
print_success "Cross-scenario config generated!"
echo

# 5. D1NAMO dataset training
print_info "Generating D1NAMO dataset training configs..."
$CONFIG_GEN --mode train --dataset d1namo --patients $PATIENTS --models $MODEL --seeds $SEED  
print_success "D1NAMO training config generated!"
echo

print_header "ALL CONFIGS GENERATED SUCCESSFULLY!"
echo "Check the experiments/ folder for your generated configurations."
echo ""
echo "Next steps:"
echo "1. Review the generated config files"
echo "2. Run your actual training/inference with these configs"
echo "3. Analyze results across different scenarios"