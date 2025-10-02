#!/bin/bash

# ==============================================================================
# Quick Config Generation Commands
# ==============================================================================
# This script contains ready-to-run config generation commands.
# Uncomment the lines you want to execute.
# ==============================================================================

CONFIG_GEN="python3 config_generator_chronos.py"
PATIENTS="570,584"
MODEL="amazon/chronos-t5-tiny"
SEED="831363"

echo "🚀 Quick Config Generation Commands"
echo "Uncomment the lines you want to run..."
echo

# =============================================================================
# TRAINING CONFIGS
# =============================================================================

echo "📚 TRAINING CONFIGS"

# Basic training on clean data (default: ohiot1dm)
# $CONFIG_GEN --mode train --patients $PATIENTS --models $MODEL --seeds $SEED

# Training on different data scenarios - OhioT1DM
# $CONFIG_GEN --mode train --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --data_scenario denoised --patients $PATIENTS --models $MODEL --seeds $SEED

# Training on different data scenarios - D1NAMO
# $CONFIG_GEN --mode train --dataset d1namo --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --dataset d1namo --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --dataset d1namo --data_scenario missing_periodic --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --dataset d1namo --data_scenario missing_random --patients $PATIENTS --models $MODEL --seeds $SEED

# =============================================================================
# PRETRAINED INFERENCE CONFIGS
# =============================================================================

echo "🔮 PRETRAINED INFERENCE CONFIGS"

# Basic inference with pretrained models
# $CONFIG_GEN --mode inference --patients $PATIENTS --models $MODEL --seeds $SEED

# Inference on different scenarios with pretrained models
# $CONFIG_GEN --mode inference --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode inference --dataset d1namo --data_scenario missing_periodic --patients $PATIENTS --models $MODEL --seeds $SEED

# =============================================================================
# TRAINED MODEL INFERENCE (SAME SCENARIO)
# =============================================================================

echo "🎯 TRAINED MODEL INFERENCE (Same Scenario)"

# Inference using your trained models (same data scenario as training)
# $CONFIG_GEN --mode trained_inference --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --dataset d1namo --data_scenario missing_periodic --patients $PATIENTS --models $MODEL --seeds $SEED

# =============================================================================
# CROSS-SCENARIO ROBUSTNESS TESTING
# =============================================================================

echo "🔄 CROSS-SCENARIO ROBUSTNESS TESTING"

# Train on clean, test on challenging data
# $CONFIG_GEN --mode trained_inference --data_scenario missing_periodic --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --data_scenario noisy --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --data_scenario missing_random --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

# Train on noisy, test on denoised (denoising effectiveness)
# $CONFIG_GEN --mode trained_inference --data_scenario denoised --train_data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED

# Cross-dataset robustness
# $CONFIG_GEN --mode trained_inference --dataset ohiot1dm --data_scenario missing_random --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --dataset d1namo --data_scenario noisy --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

# =============================================================================
# LORA INFERENCE CONFIGS
# =============================================================================

echo "🧬 LORA INFERENCE CONFIGS"

# LoRA inference (same scenario)
# $CONFIG_GEN --mode lora_inference --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode lora_inference --data_scenario noisy --patients $PATIENTS --models $MODEL --seeds $SEED

# LoRA cross-scenario robustness
# $CONFIG_GEN --mode lora_inference --data_scenario missing_periodic --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode lora_inference --data_scenario noisy --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

# =============================================================================
# RESEARCH WORKFLOWS
# =============================================================================

echo "🔬 COMPLETE RESEARCH WORKFLOWS"

# Full robustness study workflow
echo "# Full Robustness Study:"
echo "# 1. Train baseline model"
# $CONFIG_GEN --mode train --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

echo "# 2. Test on all challenging scenarios"
# $CONFIG_GEN --mode trained_inference --data_scenario missing_periodic --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --data_scenario missing_random --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --data_scenario noisy --train_data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

# Dataset comparison workflow
echo "# Dataset Comparison Study:"
# $CONFIG_GEN --mode train --dataset ohiot1dm --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode train --dataset d1namo --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --dataset ohiot1dm --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED
# $CONFIG_GEN --mode trained_inference --dataset d1namo --data_scenario standardized --patients $PATIENTS --models $MODEL --seeds $SEED

echo ""
echo "🎯 TO USE THIS SCRIPT:"
echo "1. Edit this file and uncomment the lines you want to run"
echo "2. Modify PATIENTS, MODEL, SEED variables at the top as needed"
echo "3. Run: ./quick_config_commands.sh"
echo "4. Or copy-paste individual commands to terminal"
echo ""
echo "💡 TIP: You can run multiple commands by uncommenting several lines!"