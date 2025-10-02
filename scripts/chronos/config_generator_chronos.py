#!/usr/bin/env python3
"""
Unified Chronos Configuration Generator

This script combines all chronos config generators into one unified tool with different modes:
- train: Generate training configurations  
- inference: Generate inference configurations with pretrained models
- trained_inference: Generate inference configurations using trained checkpoints
- lora_inference: Generate inference configurations using trained LoRA checkpoints

Supports multiple datasets:
- standardized: Legacy format (default)
- d1namo: D1NAMO dataset
- ohiot1dm: OhioT1DM dataset

Supports multiple data scenarios:
- standardized: Clean/raw data (default)
- noisy: Data with added noise
- denoised: Data that has been denoised
- missing_periodic: Data with periodic missing values
- missing_random: Data with random missing values

Usage:
    python config_generator_chronos.py --mode train
    python config_generator_chronos.py --mode train --dataset d1namo
    python config_generator_chronos.py --mode train --dataset ohiot1dm --data_scenario noisy
    python config_generator_chronos.py --mode trained_inference --dataset d1namo --data_scenario missing_periodic
    
Options:
    --mode: Operation mode (train, inference, trained_inference, lora_inference)
    --dataset: Dataset type (standardized, d1namo, ohiot1dm)
    --data_scenario: Data type (standardized, noisy, denoised, missing_periodic, missing_random)
    --patients: Comma-separated list of patient IDs (default: 570,584)
    --models: Comma-separated list of models (default: amazon/chronos-t5-tiny,amazon/chronos-t5-base)
    --seeds: Comma-separated list of seeds (default: from utilities.seeds)
    --output_dir: Output directory (default: auto-generated based on mode, dataset, and data_scenario)
    --help: Show this help message
"""

import os
import sys
import json
import glob
import argparse
from itertools import product
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.seeds import fixed_seeds


def find_latest_checkpoint(training_folder_pattern, patient_id):
    """
    Finds the latest 'checkpoint-final' inside the correct patient folder.
    Example path:
    experiments/chronos_training/seed_*_model_amazon_chronos-t5-*/patient_575/logs/logs_YYYY-MM-DD_HH-MM-SS/chronos-t5-*/run-0/checkpoint-final
    """
    print(f"\n🔍 Searching for checkpoint in: {training_folder_pattern} for patient {patient_id}")

    try:
        # Expand wildcard pattern (match any seed)
        training_folders = glob.glob(training_folder_pattern)
        if not training_folders:
            print(f"⚠️ No matching training folders found for {training_folder_pattern}!")
            return ""

        # Find all checkpoint-final files in the correct patient logs
        checkpoint_paths = []
        for training_folder in training_folders:
            patient_logs_path = os.path.join(training_folder, f"patient_{patient_id}")
            checkpoints = glob.glob(
                os.path.join(patient_logs_path, "**", "checkpoint-final"),
                recursive=True
            )
            if checkpoints:
                checkpoint_paths.extend(checkpoints)

        if not checkpoint_paths:
            print(f"⚠️ No 'checkpoint-final' found for patient {patient_id} in {training_folder_pattern}!")
            return ""

        # Sort checkpoints by modification time (latest first)
        checkpoint_paths.sort(key=os.path.getmtime, reverse=True)

        latest_checkpoint = checkpoint_paths[0]  # Pick the most recent one
        print(f"✅ Found latest checkpoint for patient {patient_id}: {latest_checkpoint}\\n")

        return latest_checkpoint

    except Exception as e:
        print(f"❌ Error finding checkpoint for patient {patient_id} in {training_folder_pattern}: {e}")
        return ""


def get_feature_label_sets(mode):
    """Get appropriate feature/label sets based on mode."""
    if mode == "train":
        return [{
            "input_features": ["target"],
            "labels": ["target"], 
            "prediction_length": 64,
            "context_length": 512,
        }]
    else:  # inference modes
        return [{
            "input_features": [
                "BG_{t-5}", "BG_{t-4}", "BG_{t-3}",
                "BG_{t-2}", "BG_{t-1}", "BG_{t}"
            ],
            "labels": [
                "BG_{t+1}", "BG_{t+2}", "BG_{t+3}",
                "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"
            ],
            "prediction_length": 6,
            "context_length": 6,
        }, {
            "input_features": [
                "BG_{t-8}", "BG_{t-7}", "BG_{t-6}",
                "BG_{t-5}", "BG_{t-4}", "BG_{t-3}",
                "BG_{t-2}", "BG_{t-1}", "BG_{t}"
            ],
            "labels": [
                "BG_{t+1}", "BG_{t+2}", "BG_{t+3}",
                "BG_{t+4}", "BG_{t+5}", "BG_{t+6}",
                "BG_{t+7}", "BG_{t+8}", "BG_{t+9}"
            ],
            "prediction_length": 9,
            "context_length": 9,
        }]


def get_max_train_steps(model):
    """Get max training steps based on model."""
    max_steps_map = {
        "amazon/chronos-t5-tiny": 2000,
        "amazon/chronos-t5-mini": 10000,
        "amazon/chronos-t5-small": 10000, 
        "amazon/chronos-t5-base": 2000,
        "amazon/chronos-t5-large": 1000,
        "google/t5-efficient-tiny": 2000,
        "google/t5-efficient-small": 10000,
        "google/t5-efficient-base": 2000,
    }
    return max_steps_map.get(model, 200000)


def get_data_file_path(mode, patient_id, data_scenario="standardized", dataset="ohiot1dm", prediction_length=None, context_length=None):
    """Get appropriate data file path based on mode, data scenario, and dataset."""
    
    if mode == "train":
        # For training: Use .arrow files from standardized folders
        if data_scenario == "standardized":
            # For backwards compatibility with archived generator
            if dataset == "ohiot1dm":
                return f"/home/amma/LLM-TIME/data/standardized/{patient_id}-ws-training.arrow"
            else:
                return f"/home/amma/LLM-TIME/data/{dataset}/raw_standardized/{patient_id}-ws-training.arrow"
        else:
            # For other scenarios, use the scenario-specific folder
            if dataset == "ohiot1dm":
                return f"/home/amma/LLM-TIME/data/{data_scenario}/{patient_id}-ws-training.arrow"
            else:
                return f"/home/amma/LLM-TIME/data/{dataset}/{data_scenario}/{patient_id}-ws-training.arrow"
    else:
        # For inference: Use formatted data with specific files (matches archived generators)
        # Format: ./data/formatted/{context_len}_{pred_len}
        return f"./data/formatted/{context_length}_{prediction_length}"


def generate_config_content(mode, seed, model, torch_dtype, feature_set, patient_id, 
                          max_steps=None, checkpoint_path=None, use_lora=False, data_scenario="standardized", dataset="ohiot1dm", log_folder="./logs/"):
    """Generate the configuration content based on parameters."""
    
    pred_len = feature_set["prediction_length"]
    context_len = feature_set["context_length"] 
    data_folder = get_data_file_path(mode, patient_id, data_scenario, dataset, pred_len, context_len)
    
    # Handle data paths based on mode (matches archived generators exactly)
    if mode == 'train':
        # Training mode: data_folder is the full .arrow file path
        train_data_path = data_folder
        test_data_path = f"./data/standardized/{patient_id}-ws-testing.csv"
    else:
        # Inference mode: data_folder is the base folder, append specific files
        train_data_path = f"{data_folder}/{patient_id}-ws-training.csv"
        test_data_path = f"{data_folder}/{patient_id}-ws-testing.csv"
    
    # Prepare .gin configuration content in the correct format
    config_content = f'''run.log_dir = "{log_folder}"
run.chronos_dir = "/home/amma/LLM-TIME/models/"

run.data_settings = {{
    'path_to_train_data': '{train_data_path}',
    'path_to_test_data': '{test_data_path}',
    'input_features': {feature_set["input_features"]},
    'labels': {feature_set["labels"]},
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'percent': 100
}}

run.llm_settings = {{
    'mode': '{"training" if mode == "train" else "inference"}',    
    'method': 'chronos',    
    'model': '{model}',  
    'torch_dtype': '{torch_dtype}',   
    'ntokens': 4096,
    'tokenizer_kwargs': "{{'low_limit': -30,'high_limit': 30}}",
    'prediction_length': {pred_len},    
    'num_samples': 20,
    'context_length': {context_len},
    'min_past': 60,
    'learning_rate': 0.001,
    'max_train_steps': {max_steps or get_max_train_steps(model)},
    'save_steps': 1000,
    'log_steps': 200,
    'train_batch_size': 8,
    'random_init': False,
    'seed': {seed}'''
    
    # Add checkpoint restoration settings if needed
    if mode in ['trained_inference', 'lora_inference'] and checkpoint_path:
        config_content += f''',
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{checkpoint_path}' '''
    
    # Add LoRA settings
    if use_lora:
        config_content += f''',
    'use_peft': True,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05'''
    else:
        config_content += f''',
    'use_peft': False,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05'''
    
    config_content += '''
}'''
    
    return config_content
    
    return "\n".join(gin_content)


def main():
    parser = argparse.ArgumentParser(description="Unified Chronos Configuration Generator")
    parser.add_argument("--mode", required=True, 
                       choices=["train", "inference", "trained_inference", "lora_inference"],
                       help="Operation mode")
    parser.add_argument("--patients", default="570,584", 
                       help="Comma-separated patient IDs")
    parser.add_argument("--models", default="amazon/chronos-t5-tiny,amazon/chronos-t5-base",
                       help="Comma-separated model names") 
    parser.add_argument("--seeds", default=None,
                       help="Comma-separated seeds (default: use fixed_seeds)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--data_scenario", default="standardized",
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Data scenario type (default: standardized)")
    parser.add_argument("--dataset", default="ohiot1dm",
                       choices=["ohiot1dm", "d1namo"],
                       help="Dataset type (default: ohiot1dm)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Enable LoRA (PEFT) for training mode")
    
    args = parser.parse_args()
    
    # Parse inputs
    patients = [p.strip() for p in args.patients.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds
    
    # Set output directory based on mode
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Include dataset and scenario in directory names
        dataset_suffix = "" if args.dataset == "ohiot1dm" else f"_{args.dataset}"
        scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
        combined_suffix = f"{dataset_suffix}{scenario_suffix}"
        
        dir_map = {
            "train": f"./experiments/chronos_training{combined_suffix}/",
            "inference": f"./experiments/chronos_inference{combined_suffix}/", 
            "trained_inference": f"./experiments/chronos_trained_inference{combined_suffix}/",
            "lora_inference": f"./experiments/chronos_lora_inference{combined_suffix}/"
        }
        base_output_dir = dir_map[args.mode]
    
    # Get feature sets and other parameters
    feature_label_sets = get_feature_label_sets(args.mode)
    torch_dtypes = ["float32"]
    
    print(f"🚀 Starting {args.mode} config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"� Data scenario: {args.data_scenario}")
    print(f"�👥 Patients: {patients}")
    print(f"🤖 Models: {models}")
    print(f"🎲 Seeds: {seeds}")
    
    # Generate configurations
    config_count = 0
    for seed, feature_set, model, torch_dtype in product(seeds, feature_label_sets, models, torch_dtypes):
        
        pred_len = feature_set["prediction_length"]
        context_len = feature_set["context_length"]
        
        # Create experiment folder
        folder_name = f"seed_{seed}_model_{model.replace('/', '-').replace('_','-')}_dtype_{torch_dtype}_mode_{args.mode}_context_{context_len}_pred_{pred_len}"
        experiment_folder = os.path.join(base_output_dir, folder_name)
        
        for patient_id in patients:
            patient_folder = os.path.join(experiment_folder, f"patient_{patient_id}")
            log_folder = os.path.join(patient_folder, "logs")
            
            # Create directories
            os.makedirs(patient_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)
            
            # Handle checkpoint finding for trained inference modes
            checkpoint_path = None
            if args.mode in ["trained_inference", "lora_inference"]:
                # Use same dataset and scenario suffix for finding training checkpoints
                dataset_suffix = "" if args.dataset == "standardized" else f"_{args.dataset}"
                scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
                combined_suffix = f"{dataset_suffix}{scenario_suffix}"
                training_pattern = f"./experiments/chronos_training{combined_suffix}/seed_{seed}_model_{model.replace('/', '-').replace('_','-')}_dtype_{torch_dtype}_mode_training_*"
                checkpoint_path = find_latest_checkpoint(training_pattern, patient_id)
                if not checkpoint_path:
                    print(f"⚠️ Skipping {patient_id} - no checkpoint found")
                    continue
            
            # Generate config content
            use_lora = (args.mode == "lora_inference") or (args.mode == "train" and getattr(args, 'use_lora', False))
            config_content = generate_config_content(
                args.mode, seed, model, torch_dtype, feature_set, patient_id,
                checkpoint_path=checkpoint_path, use_lora=use_lora, data_scenario=args.data_scenario, dataset=args.dataset, log_folder=log_folder
            )
            
            # Save config file
            config_path = os.path.join(patient_folder, "config.gin")
            with open(config_path, "w") as f:
                f.write(config_content)
            
            print(f"Generated: {config_path}")
            config_count += 1
    
    print(f"✅ Generated {config_count} configuration files in {base_output_dir}")


if __name__ == "__main__":
    main()