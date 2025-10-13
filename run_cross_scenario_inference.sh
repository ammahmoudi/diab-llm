# #!/bin/bash

# # Cross-Scenario Inference Script for Chronos
# # Generates inference configs for denoised and noisy data using trained checkpoints from standardized data
# # Then runs the inference experiments

# set -e  # Exit on error

# echo "🚀 Starting Cross-Scenario Inference Generation and Execution"
# echo "============================================================="
# echo "Training scenario: standardized"
# echo "Test scenarios: denoised, noisy"
# echo "Patients: All 12 OhioT1DM patients"
# echo "Seeds: All 5 seeds"
# echo "Window config: 6_6"
# echo ""

# # Activate virtual environment
# source venv/bin/activate

# # 1. Generate inference configs for denoised data (trained on standardized)
# # echo "📋 Step 1/4: Generating inference configs for DENOISED data..."
# # python scripts/chronos/config_generator_chronos.py \
# #     --mode trained_inference \
# #     --dataset ohiot1dm \
# #     --data_scenario denoised \
# #     --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
# #     --models amazon/chronos-t5-base \
# #     --seeds 831363,809906,427368,238822,247659 \
# #     --train_scenario standardized \
# #     --window_config 6_6

# # echo "✅ Denoised inference configs generated successfully!"
# # echo ""

# # 2. Generate inference configs for noisy data (trained on standardized)
# echo "📋 Step 2/4: Generating inference configs for NOISY data..."
# python scripts/chronos/config_generator_chronos.py \
#     --mode trained_inference \
#     --dataset ohiot1dm \
#     --data_scenario noisy \
#     --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
#     --models amazon/chronos-t5-base \
#     --seeds 831363,809906,427368,238822,247659 \
#     --train_scenario standardized \
#     --window_config 6_6

# echo "✅ Noisy inference configs generated successfully!"
# echo ""

# # 3. Run inference experiments for denoised data
# echo "🎯 Step 3/4: Running inference experiments for DENOISED data..."
# python scripts/chronos/run_all_chronos_experiments.py \
#     --modes trained_inference \
#     --datasets ohiot1dm \
#     --fix_results

# echo "✅ Denoised inference experiments completed!"
# echo ""

# # 4. Run inference experiments for noisy data
# echo "🎯 Step 4/4: Running inference experiments for NOISY data..."
# python scripts/chronos/run_all_chronos_experiments.py \
#     --modes trained_inference \
#     --datasets ohiot1dm \
#     --fix_results

# echo "✅ Noisy inference experiments completed!"
# echo ""

# # Summary
# echo "🎉 All cross-scenario inference experiments completed successfully!"
# echo "============================================================="
# echo "Results should be available in:"
# echo "- chronos_trained_inference_ohiot1dm_denoised_results.csv"
# echo "- chronos_trained_inference_ohiot1dm_noisy_results.csv"
# echo ""
# echo "Experiment folders:"
# echo "- ./experiments/chronos_trained_inference_ohiot1dm_denoised/"
# echo "- ./experiments/chronos_trained_inference_ohiot1dm_noisy/"





#!/bin/bash

# Cross-Scenario Train+Inference Script for Time-LLM with BERT (6_6 window only)
# Generates train_inference configs for denoised and noisy data using BERT
# Uses train_inference mode which combines training and inference in one step

set -e  # Exit on error

echo "🚀 Starting Time-LLM BERT Cross-Scenario Train+Inference (6_6 window)"
echo "======================================================================"
echo "Model: Time-LLM with BERT"
echo "Mode: train_inference (combined)"
echo "Test scenarios: denoised, noisy"
echo "Patients: All 12 OhioT1DM patients"
echo "Seeds: All 5 seeds"
echo "Window config: 6_6 only"
echo ""

# Activate virtual environment
source venv/bin/activate

# 1. Generate train_inference configs for denoised data (train on standardized, test on denoised)
echo "📋 Step 1/4: Generating Time-LLM BERT train_inference configs for DENOISED data (6_6)..."
python scripts/time_llm/config_generator_time_llm_unified.py \
    --mode train_inference \
    --dataset ohiot1dm \
    --data_scenario denoised \
    --train_data_scenario standardized \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT \
    --seeds 831363,809906,427368,238822,247659

echo "✅ Time-LLM BERT denoised train_inference configs generated successfully!"
echo ""

# 2. Generate train_inference configs for noisy data (train on standardized, test on noisy)
echo "📋 Step 2/4: Generating Time-LLM BERT train_inference configs for NOISY data (6_6)..."
python scripts/time_llm/config_generator_time_llm_unified.py \
    --mode train_inference \
    --dataset ohiot1dm \
    --data_scenario noisy \
    --train_data_scenario standardized \
    --patients 540,544,552,559,563,567,570,575,584,588,591,596 \
    --llm_models BERT \
    --seeds 831363,809906,427368,238822,247659

echo "✅ Time-LLM BERT noisy train_inference configs generated successfully!"
echo ""

# 3. Run train_inference experiments for denoised data
echo "🎯 Step 3/4: Running Time-LLM BERT train_inference experiments for DENOISED data..."
python scripts/time_llm/run_all_time_llm_experiments.py \
    --modes train_inference \
    --datasets ohiot1dm

echo "✅ Time-LLM BERT denoised train_inference experiments completed!"
echo ""

# 4. Run train_inference experiments for noisy data
echo "🎯 Step 4/4: Running Time-LLM BERT train_inference experiments for NOISY data..."
python scripts/time_llm/run_all_time_llm_experiments.py \
    --modes train_inference \
    --datasets ohiot1dm

echo "✅ Time-LLM BERT noisy train_inference experiments completed!"
echo ""

# Summary
echo "🎉 All Time-LLM BERT train_inference experiments completed successfully!"
echo "======================================================================"
echo "Results should be available in:"
echo "- time_llm_train_inference_ohiot1dm_denoised_results.csv"
echo "- time_llm_train_inference_ohiot1dm_noisy_results.csv"
echo ""
echo "Experiment folders:"
echo "- ./experiments/time_llm_train_inference_ohiot1dm_denoised/"
echo "- ./experiments/time_llm_train_inference_ohiot1dm_noisy/"