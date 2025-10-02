#!/usr/bin/env python3
"""
Unified Chronos Configuration Generator

This script combines all chronos config generators into one unified tool with different modes:
- train: Generate training configurations  
- inference: Generate inference configurations with pretrained models
- trained_inference: Generate inference configurations using trained checkpoints
- lora_inference: Generate inference configurations using trained LoRA checkpoints

Supports multiple datasets:
- ohiot1dm: OhioT1DM dataset (default)
- d1namo: D1NAMO dataset

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
    
    # Cross-scenario evaluation (train on clean, test on missing data):
    python config_generator_chronos.py --mode trained_inference --data_scenario missing_periodic --train_data_scenario standardized
    
Options:
    --mode: Operation mode (train, inference, trained_inference, lora_inference)
    --dataset: Dataset type (ohiot1dm, d1namo)
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
        "amazon/chronos-t5-tiny": 200000,
        "amazon/chronos-t5-small": 200000, 
        "amazon/chronos-t5-base": 200000,
        "amazon/chronos-t5-large": 200000,
        "google/t5-efficient-tiny": 200000,
        "google/t5-efficient-small": 200000,
        "google/t5-efficient-base": 200000,
    }
    return max_steps_map.get(model, 200000)


def get_data_file_path(mode, patient_id, data_scenario="standardized", dataset="ohiot1dm"):
    """Get appropriate data file path based on mode, data scenario, and dataset."""
    
    # Map scenarios to subfolder names in datasets
    scenario_map = {
        "standardized": "raw_standardized",  
        "noisy": "noisy",
        "denoised": "denoised", 
        "missing_periodic": "missing_periodic",
        "missing_random": "missing_random"
    }
    
    scenario_folder = scenario_map[data_scenario]
    base_path = f"/home/amma/LLM-TIME/data/{dataset}/{scenario_folder}"
    
    if mode == "train":
        return f"{base_path}/{patient_id}-ws-training.arrow"
    else:
        # For inference, might need formatted data
        if data_scenario != "standardized":
            return f"./data/{dataset}/{scenario_folder}_formatted"
        else:
            return f"{base_path}/{patient_id}-ws-test.arrow"


def generate_config_content(mode, seed, model, torch_dtype, feature_set, patient_id, 
                          max_steps=None, checkpoint_path=None, use_lora=False, data_scenario="standardized", dataset="ohiot1dm"):
    """Generate the configuration content based on parameters."""
    
    pred_len = feature_set["prediction_length"]
    context_len = feature_set["context_length"] 
    data_folder = get_data_file_path(mode, patient_id, data_scenario, dataset)
    
    # Base configuration
    config = {
        'mode': 'training' if mode == 'train' else 'inference',
        'random_seed': seed,
        'model_name': model,
        'data_folder': data_folder,
        'input_features': feature_set["input_features"],
        'labels': feature_set["labels"],
        'torch_dtype': torch_dtype,
        'prediction_length': pred_len,
        'context_length': context_len,
        'restore_from_checkpoint': False,
        'restore_checkpoint_path': '',
    }
    
    # Mode-specific configurations
    if mode == 'train':
        config.update({
            'max_steps': max_steps or get_max_train_steps(model),
            'use_lora': True,
            'lora_r': 16,
            'lora_alpha': 32, 
            'lora_dropout': 0.05
        })
    elif mode in ['trained_inference', 'lora_inference']:
        if checkpoint_path:
            config.update({
                'restore_from_checkpoint': True,
                'restore_checkpoint_path': checkpoint_path,
            })
            if use_lora:
                config.update({
                    'use_lora': True,
                    'lora_r': 16,
                    'lora_alpha': 32,
                    'lora_dropout': 0.05
                })
    
    # Convert to gin format
    gin_content = []
    for key, value in config.items():
        if isinstance(value, str):
            gin_content.append(f"{key} = '{value}'")
        elif isinstance(value, list):
            formatted_list = str(value).replace("'", '"')
            gin_content.append(f"{key} = {formatted_list}")
        else:
            gin_content.append(f"{key} = {value}")
    
    return "\\n".join(gin_content)


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
    parser.add_argument("--train_data_scenario", default=None,
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Training data scenario (for trained_inference/lora_inference modes). If not specified, uses --data_scenario")
    
    args = parser.parse_args()
    
    # Parse inputs
    patients = [p.strip() for p in args.patients.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds
    
    # Handle cross-scenario evaluation
    train_scenario = args.train_data_scenario if args.train_data_scenario else args.data_scenario
    inference_scenario = args.data_scenario
    
    # Set output directory based on mode
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # Include dataset and scenario in directory names
        dataset_suffix = "" if args.dataset == "ohiot1dm" else f"_{args.dataset}"
        
        if args.mode == "train":
            # For training, only use the data scenario
            scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
        elif args.mode in ["trained_inference", "lora_inference"] and args.train_data_scenario:
            # For cross-scenario inference, show both scenarios
            scenario_suffix = f"_train_{train_scenario}_test_{inference_scenario}" if train_scenario != inference_scenario else f"_{inference_scenario}"
        else:
            # Default behavior
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
    print(f"🗃️  Dataset: {args.dataset}")
    if args.mode in ["trained_inference", "lora_inference"] and args.train_data_scenario:
        print(f"🏋️  Training scenario: {train_scenario}")
        print(f"📊 Inference scenario: {inference_scenario}")
    else:
        print(f"📊 Data scenario: {args.data_scenario}")
    print(f"👥 Patients: {patients}")
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
                # Use training scenario for finding checkpoints, not inference scenario
                dataset_suffix = "" if args.dataset == "ohiot1dm" else f"_{args.dataset}"
                train_scenario_suffix = "" if train_scenario == "standardized" else f"_{train_scenario}"
                training_combined_suffix = f"{dataset_suffix}{train_scenario_suffix}"
                training_pattern = f"./experiments/chronos_training{training_combined_suffix}/seed_{seed}_model_{model.replace('/', '-').replace('_','-')}_dtype_{torch_dtype}_mode_training_*"
                checkpoint_path = find_latest_checkpoint(training_pattern, patient_id)
                if not checkpoint_path:
                    print(f"⚠️ Skipping {patient_id} - no checkpoint found")
                    continue
            
            # Generate config content
            use_lora = (args.mode in ["train", "lora_inference"])
            # Use inference scenario for data path (what we're testing on)
            data_scenario_for_inference = inference_scenario
            config_content = generate_config_content(
                args.mode, seed, model, torch_dtype, feature_set, patient_id,
                checkpoint_path=checkpoint_path, use_lora=use_lora, data_scenario=data_scenario_for_inference, dataset=args.dataset
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