#!/usr/bin/env python3
"""
Unified Time-LLM Configuration Generator

This script combines all time-llm config generators into one unified tool with different modes:
- train: Generate training configurations  
- inference: Generate inference configurations (epochs=0)
- train_inference: Generate combined training+inference configurations

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
    python config_generator_time_llm.py --mode train
    python config_generator_time_llm.py --mode train --dataset d1namo
    python config_generator_time_llm.py --mode train --dataset ohiot1dm --data_scenario noisy
    python config_generator_time_llm.py --mode inference --dataset d1namo --data_scenario missing_periodic
    
    # Cross-scenario evaluation (train on clean, test on missing data):
    python config_generator_time_llm.py --mode train_inference --data_scenario missing_periodic --train_data_scenario standardized
    
Options:
    --mode: Operation mode (train, inference, train_inference)
    --dataset: Dataset type (ohiot1dm, d1namo)
    --data_scenario: Data type for inference/testing (standardized, noisy, denoised, missing_periodic, missing_random)
    --train_data_scenario: Data type used for training (optional, for cross-scenario evaluation)
    --patients: Comma-separated list of patient IDs (default: 570,584)
    --llm_models: Comma-separated list of LLM models (default: GPT2,LLAMA)
    --seeds: Comma-separated list of seeds (default: from utilities.seeds)
    --epochs: Number of training epochs (default: 10 for train, 0 for inference)
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


def get_llm_config(llm_model):
    """Get LLM model configuration."""
    llm_configs = {
        "GPT2": {"llm_model": "GPT2", "llm_dim": 768},
        "LLAMA": {"llm_model": "LLAMA", "llm_dim": 4096},
        "BERT": {"llm_model": "BERT", "llm_dim": 768},
    }
    return llm_configs.get(llm_model, llm_configs["GPT2"])


def get_length_sets(mode):
    """Get appropriate length sets based on mode."""
    # Common length configurations for Time-LLM
    return [
        {"sequence_length": 6, "context_length": 6, "prediction_length": 6, "patch_len": 6},
        {"sequence_length": 6, "context_length": 6, "prediction_length": 9, "patch_len": 6},
    ]


def get_data_file_path(mode, patient_id, data_scenario="standardized", dataset="ohiot1dm", is_train_data=True):
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
    base_path = f"./data/{dataset}/{scenario_folder}"
    
    # For Time-LLM, we use CSV files
    if is_train_data:
        return f"{base_path}/{patient_id}-ws-training.csv"
    else:
        return f"{base_path}/{patient_id}-ws-testing.csv"


def generate_config_content(mode, seed, llm_config, length_set, patient_id, train_epochs=10,
                          data_scenario="standardized", dataset="ohiot1dm", train_data_scenario=None):
    """Generate the configuration content based on parameters."""
    
    # Use train_data_scenario for training data, data_scenario for test data
    actual_train_scenario = train_data_scenario if train_data_scenario else data_scenario
    
    train_data_path = get_data_file_path(mode, patient_id, actual_train_scenario, dataset, is_train_data=True)
    test_data_path = get_data_file_path(mode, patient_id, data_scenario, dataset, is_train_data=False)
    
    # Get prompt file path
    prompt_path = f"./data/{dataset}/{scenario_map.get(data_scenario, 'raw_standardized')}/t1dm_prompt.txt"
    
    log_folder_placeholder = "LOGS_PLACEHOLDER"  # Will be replaced with actual log folder
    
    config_content = f'''run.log_dir = "{log_folder_placeholder}"
run.data_settings = {{
    'path_to_train_data': '{train_data_path}',
    'path_to_test_data': '{test_data_path}',
    'input_features': ['target'],
    'labels': ['target'],
    'prompt_path': '{prompt_path}',
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'frequency': '5min',
    'percent': 100,
    'random_seed': {seed}
}}

run.training_settings = {{
    'random_seed': {seed},
    'train_epochs': {train_epochs},
    'batch_size': 24,
    'learning_rate': 0.01,
    'des': 'Experiment',
    'itr': 1,
    'patience': 10,
    'lradj': 'type1',
    'use_amp': False,
    'comment': 'none',
    'model_comment': 'time_llm_{llm_config["llm_model"]}_{llm_config["llm_dim"]}_{length_set["sequence_length"]}_{length_set["context_length"]}_{length_set["prediction_length"]}_{length_set["patch_len"]}',
    'model_id': 'test',
    'model': 'TimeLLM',
    'seq_len': {length_set["sequence_length"]},
    'label_len': 0,
    'pred_len': {length_set["prediction_length"]},
    'd_model': 32,
    'd_ff': 128,
    'patch_len': {length_set["patch_len"]},
    'stride': {length_set["patch_len"]},
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'llm_model': '{llm_config["llm_model"]}',
    'llm_dim': {llm_config["llm_dim"]},
    'llm_layers': 6
}}

run.hardware_settings = {{
    'use_gpu': True,
    'gpu': 0,
    'use_multi_gpu': False,
    'devices': '0,1,2,3',
    'p_hidden_dims': [128, 128],
    'p_hidden_layers': 2
}}'''
    
    return config_content


def main():
    parser = argparse.ArgumentParser(description="Unified Time-LLM Configuration Generator")
    parser.add_argument("--mode", required=True, 
                       choices=["train", "inference", "train_inference"],
                       help="Operation mode")
    parser.add_argument("--patients", default="570,584", 
                       help="Comma-separated patient IDs")
    parser.add_argument("--llm_models", default="GPT2,LLAMA",
                       help="Comma-separated LLM model names") 
    parser.add_argument("--seeds", default=None,
                       help="Comma-separated seeds (default: use fixed_seeds)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (default: 10 for train modes, 0 for inference)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--data_scenario", default="standardized",
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Data scenario type for inference/testing (default: standardized)")
    parser.add_argument("--dataset", default="ohiot1dm",
                       choices=["ohiot1dm", "d1namo"],
                       help="Dataset type (default: ohiot1dm)")
    parser.add_argument("--train_data_scenario", default=None,
                       choices=["standardized", "noisy", "denoised", "missing_periodic", "missing_random"],
                       help="Training data scenario (for cross-scenario evaluation). If not specified, uses --data_scenario")
    
    args = parser.parse_args()
    
    # Parse inputs
    patients = [p.strip() for p in args.patients.split(",")]
    llm_models = [m.strip() for m in args.llm_models.split(",")]
    
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds[:2]  # Use first 2 seeds by default
    
    # Set epochs based on mode
    if args.epochs is not None:
        train_epochs = args.epochs
    else:
        train_epochs = 0 if args.mode == "inference" else 10
    
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
            dir_name = "time_llm_training"
        elif args.mode == "inference":
            scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_inference"
        elif args.mode == "train_inference":
            if args.train_data_scenario:
                # Cross-scenario case
                scenario_suffix = f"_train_{train_scenario}_test_{inference_scenario}" if train_scenario != inference_scenario else f"_{inference_scenario}"
            else:
                scenario_suffix = "" if args.data_scenario == "standardized" else f"_{args.data_scenario}"
            dir_name = "time_llm_training_inference"
            
        combined_suffix = f"{dataset_suffix}{scenario_suffix}"
        base_output_dir = f"./experiments/{dir_name}{combined_suffix}/"
    
    # Get length sets
    length_sets = get_length_sets(args.mode)
    torch_dtypes = ["bfloat16"]
    model_ids = ["test"]
    
    print(f"🚀 Starting {args.mode} config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"🗃️  Dataset: {args.dataset}")
    if args.train_data_scenario:
        print(f"🏋️  Training scenario: {train_scenario}")
        print(f"📊 Inference scenario: {inference_scenario}")
    else:
        print(f"📊 Data scenario: {args.data_scenario}")
    print(f"👥 Patients: {patients}")
    print(f"🤖 LLM Models: {llm_models}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📈 Epochs: {train_epochs}")
    
    # Generate configurations
    config_count = 0
    for seed, llm_model_name, length_set, torch_dtype, model_id in product(seeds, llm_models, length_sets, torch_dtypes, model_ids):
        
        llm_config = get_llm_config(llm_model_name)
        seq_len = length_set["sequence_length"]
        context_len = length_set["context_length"]
        pred_len = length_set["prediction_length"]
        patch_len = length_set["patch_len"]
        
        # Create experiment folder
        folder_name = f"seed_{seed}_model_{llm_config['llm_model']}_dim_{llm_config['llm_dim']}_seq_{seq_len}_context_{context_len}_pred_{pred_len}_patch_{patch_len}_epochs_{train_epochs}"
        experiment_folder = os.path.join(base_output_dir, folder_name)
        
        for patient_id in patients:
            patient_folder = os.path.join(experiment_folder, f"patient_{patient_id}")
            log_folder = os.path.join(patient_folder, "logs")
            
            # Create directories
            os.makedirs(patient_folder, exist_ok=True)
            os.makedirs(log_folder, exist_ok=True)
            
            # Generate config content
            config_content = generate_config_content(
                args.mode, seed, llm_config, length_set, patient_id, train_epochs,
                data_scenario=args.data_scenario, dataset=args.dataset, train_data_scenario=args.train_data_scenario
            )
            
            # Replace log folder placeholder with actual path
            config_content = config_content.replace("LOGS_PLACEHOLDER", log_folder)
            
            # Save config file
            config_path = os.path.join(patient_folder, "config.gin")
            with open(config_path, "w") as f:
                f.write(config_content)
            
            print(f"Generated: {config_path}")
            config_count += 1
    
    print(f"✅ Generated {config_count} configuration files in {base_output_dir}")


# Add scenario_map to global scope for generate_config_content
scenario_map = {
    "standardized": "raw_standardized",  
    "noisy": "noisy",
    "denoised": "denoised", 
    "missing_periodic": "missing_periodic",
    "missing_random": "missing_random"
}


if __name__ == "__main__":
    main()