import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import import_utils

import json
from itertools import product
import random
import argparse
from utilities.seeds import fixed_seeds


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Time-LLM configuration files')
    parser.add_argument('--mode', type=str, required=True, 
                       help='Mode: train_inference, training, inference, etc.')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., ohiot1dm)')
    parser.add_argument('--data_scenario', type=str, required=True,
                       help='Data scenario: denoised, noisy, standardized, etc.')
    parser.add_argument('--patients', type=str, required=True,
                       help='Comma-separated list of patient IDs')
    parser.add_argument('--models', type=str, required=True,
                       help='Model type (e.g., bert, gpt2, llama)')
    parser.add_argument('--seeds', type=str, required=True,
                       help='Comma-separated list of seeds')
    parser.add_argument('--window_config', type=str, required=True,
                       help='Window configuration (e.g., 6_6)')
    
    return parser.parse_args()


def get_model_config(model_name):
    """Get model configuration based on model name"""
    model_configs = {
        "bert": {"llm_model": "BERT", "llm_dim": 768},
        "gpt2": {"llm_model": "GPT2", "llm_dim": 768},
        "llama": {"llm_model": "LLAMA", "llm_dim": 4096},
    }
    return model_configs.get(model_name.lower(), {"llm_model": "BERT", "llm_dim": 768})


def get_window_config(window_str):
    """Parse window configuration string like '6_6' into context and prediction lengths"""
    parts = window_str.split('_')
    if len(parts) == 2:
        context_len = int(parts[0])
        pred_len = int(parts[1])
        return {
            "sequence_length": context_len, 
            "context_length": context_len, 
            "prediction_length": pred_len, 
            "patch_len": context_len
        }
    else:
        # Default fallback
        return {
            "sequence_length": 6, 
            "context_length": 6, 
            "prediction_length": 6, 
            "patch_len": 6
        }


def main():
    args = parse_args()
    
    # Parse arguments
    patients = args.patients.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    model_config = get_model_config(args.models)
    window_config = get_window_config(args.window_config)
    
    # Set train epochs based on mode
    train_epochs = 100 if 'train' in args.mode else 0
    
    # Set output directory based on mode
    if train_epochs > 0:
        base_output_dir = f"./experiments/time_llm_{args.mode}_{args.dataset}_{args.data_scenario}"
    else:
        base_output_dir = f"./experiments/time_llm_{args.mode}_{args.dataset}_{args.data_scenario}"
    
    os.makedirs(base_output_dir, exist_ok=True)

    # Generate configurations
    llm_model = model_config["llm_model"]
    llm_dim = model_config["llm_dim"]
    
    sequence_len = window_config["sequence_length"]
    context_len = window_config["context_length"]
    pred_len = window_config["prediction_length"]
    patch_len = window_config["patch_len"]

    for seed in seeds:
        model_comment = f"time_llm_{llm_model}_{llm_dim}_{sequence_len}_{context_len}_{pred_len}_{patch_len}"
        
        config_folder = os.path.join(
            base_output_dir,
            f"seed_{seed}_model_{llm_model.lower()}_dim_{llm_dim}_seq_{sequence_len}_context_{context_len}_pred_{pred_len}_patch_{patch_len}_epochs_{train_epochs}",
        )
        os.makedirs(config_folder, exist_ok=True)

        for patient_id in patients:
            patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
            os.makedirs(patient_folder, exist_ok=True)
            
            log_folder = os.path.join(patient_folder, "logs")
            os.makedirs(log_folder, exist_ok=True)
            
            # Set data folder based on data scenario
            data_folder = f"./data/ohiot1dm/{args.data_scenario}_formatted/{args.window_config}"

            config_content = f"""
run.log_dir = "{log_folder}"
run.data_settings = {{
    'path_to_train_data': '{data_folder}/{patient_id}-ws-training.csv',
    'path_to_test_data': '{data_folder}/{patient_id}-ws-testing.csv',
    'input_features': ['target'],
    'labels': ['target'],
    'prompt_path': '{data_folder}/t1dm_prompt.txt',
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'frequency': '5min',
    'percent': 100,
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': '{args.mode}',
    'method': 'time_llm',
    'llm_model': '{llm_model}',
    'llm_layers': 8,
    'llm_dim': {llm_dim},
    'num_workers': 1,
    'torch_dtype': 'bfloat16',
    'model_id': 'test',
    'sequence_length': {sequence_len},
    'context_length': {context_len},
    'prediction_length': {pred_len},
    'patch_len': {patch_len},
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.001,
    'train_epochs': {train_epochs},
    'features': 'S',
    'd_model': 32,
    'd_ff': 32,
    'factor': 1,
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'e_layers': 2,
    'd_layers': 1,
    'n_heads': 8,
    'dropout': 0.1,
    'moving_avg': 25,
    'activation': 'gelu',
    'embed': 'timeF',
    'patience': 10,
    'lradj': 'COS',
    'des': 'test',
    'model_comment': '{model_comment}',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {seed}
}}
"""

            config_filename = "config.gin"
            config_path = os.path.join(patient_folder, config_filename)

            with open(config_path, "w") as f:
                f.write(config_content.strip())

            print(f"Generated: {config_path} with logs stored in {log_folder}")

    print(f"\nGenerated {len(seeds) * len(patients)} configuration files for {args.mode} mode")
    print(f"Dataset: {args.dataset}, Data scenario: {args.data_scenario}")
    print(f"Model: {llm_model}, Window config: {args.window_config}")
    print(f"Output directory: {base_output_dir}")


if __name__ == "__main__":
    main()
