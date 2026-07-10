#!/usr/bin/env python3
"""
MIT-BIH Time-LLM Configuration Generator

This script follows the same structure and style as the main unified
Time-LLM config generator, but targets the MIT-BIH ECG classification path.

Supported modes:
- train: Generate training configurations
- inference: Generate inference configurations
- train_inference: Generate combined training+inference configurations

Supported dataset:
- mitbih: MIT-BIH Arrhythmia dataset for beat-centered AAMI 5-class ECG classification

Usage:
    python config_generator_mitbih.py --mode train
    python config_generator_mitbih.py --mode train_inference --llm_models TinyBERT,BERT-tiny
    python config_generator_mitbih.py --mode inference --checkpoint-path <ckpt>

Options:
    --mode: Operation mode (train, inference, train_inference)
    --llm_models: Comma-separated list of LLM models
    --seeds: Comma-separated seeds (default: from utilities.seeds)
    --epochs: Number of training epochs
    --output_dir: Output directory (default: auto-generated)
    --torch-dtype: Torch dtype for model
    --checkpoint-path: Checkpoint path required for inference mode
    --include-duplicate-202: Include record 202 instead of the default curated 47-record set
    --unfreeze-llm: Fine-tune the LLM backbone instead of freezing it
"""

import os
import sys
import argparse
from itertools import product

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.seeds import fixed_seeds
from time_llm.config_generator import get_llm_config, get_model_batch_sizes


def get_length_sets(mode):
    """Get appropriate length sets for MIT-BIH ECG classification."""
    return [
        {"sequence_length": 256, "context_length": 256, "prediction_length": 0, "patch_len": 16},
        {"sequence_length": 300, "context_length": 300, "prediction_length": 0, "patch_len": 20},
    ]


def generate_config_content(mode, seed, llm_config, length_set, train_epochs=10,
                            checkpoint_path=None, torch_dtype="float32",
                            include_duplicate_202=False,
                            class_balanced=False, class_balanced_max_oversample=50.0):
    """Generate the configuration content based on parameters."""
    log_folder_placeholder = "LOGS_PLACEHOLDER"

    mode_str = "training+inference" if mode != "inference" else "inference"
    batch_sizes = get_model_batch_sizes(llm_config["llm_model"])
    restore_flag = mode == "inference" and checkpoint_path is not None

    class_balanced_line = (
        f",\n     'class_balanced_sampling': True,"
        f"\n     'class_balanced_max_oversample': {class_balanced_max_oversample}"
        if class_balanced else ""
    )

    config_content = f'''# Parameters for run:
# ==============================================================================
run.data_settings = \\
    {{'dataset_dir': './data/mit-bih-arrhythmia',
     'metadata_csv': './data/mit-bih-arrhythmia/metadata_records.csv',
     'demographics_csv': './data/mit-bih-arrhythmia/demographics_records.csv',
     'beat_index_csv': './data/mit-bih-arrhythmia/beat_index.csv',
     'include_duplicate_202': {str(include_duplicate_202)}{class_balanced_line}}}

run.llm_settings = \\
    {{'activation': 'gelu',
     'c_out': 5,
     'context_length': {length_set["context_length"]},
     'd_ff': 128,
     'd_layers': 1,
     'd_model': 64,
     'dec_in': 1,
     'des': 'mitbih',
     'dropout': 0.1,
     'e_layers': 2,
     'embed': 'timeF',
     'enc_in': 1,
     'eval_metrics': ['accuracy', 'macro_f1', 'weighted_f1'],
    'factor': 1,
     'learning_rate': 0.0001,
     'llm_dim': {llm_config["llm_dim"]},
     'llm_layers': {llm_config["llm_layers"]},
     'llm_model': '{llm_config["llm_model"]}',
     'lradj': 'COS',
     'method': 'time_llm_ecg_classifier',
     'mode': '{mode_str}',
     'model_comment': 'time_llm_ecg_{llm_config["llm_model"]}_{llm_config["llm_dim"]}_{length_set["sequence_length"]}_{length_set["patch_len"]}',
     'model_id': 'mitbih',
     'moving_avg': 25,
     'n_heads': 8,
     'num_classes': 5,
     'num_workers': 0,
     'patch_len': {length_set["patch_len"]},
     'patience': 10,
     'pooling': 'mean',
     'prediction_batch_size': {batch_sizes["prediction_batch_size"]},
     'prediction_length': {length_set["prediction_length"]},
     'prompt_domain': 0,
     'restore_from_checkpoint': {str(restore_flag)},
     'restore_checkpoint_path': '{checkpoint_path or ''}',
     'seed': {seed},
     'sequence_length': {length_set["sequence_length"]},
     'stride': 8,
     'task_name': 'ecg_classification',
     'timeenc': 0,
     'torch_dtype': '{torch_dtype}',
     'train_batch_size': {batch_sizes["train_batch_size"]},
     'train_epochs': {train_epochs}}}
run.log_dir = \\
    '{log_folder_placeholder}'
'''

    return config_content


def main():
    parser = argparse.ArgumentParser(description="MIT-BIH Time-LLM Configuration Generator")
    parser.add_argument("--mode", required=True,
                       choices=["train", "inference", "train_inference"],
                       help="Operation mode (train, inference, train_inference)")
    parser.add_argument("--llm_models", default="TinyBERT,BERT-tiny",
                       help="Comma-separated LLM model names")
    parser.add_argument("--seeds", default=None,
                       help="Comma-separated seeds (default: use fixed_seeds)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (default: 10 for train modes, 0 for inference)")
    parser.add_argument("--output_dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--torch-dtype", default="float32",
                       choices=["float32", "bfloat16", "float16"],
                       help="Torch dtype for model (default: float32)")
    parser.add_argument("--checkpoint-path", default=None,
                       help="Path to checkpoint for inference mode (required for inference)")
    parser.add_argument("--include-duplicate-202", action="store_true",
                       help="Include duplicate record 202 instead of using the curated 47-record set")
    parser.add_argument("--class-balanced", action="store_true",
                       help="Enable class-balanced oversampling for training (recommended: MIT-BIH's "
                            "extreme AAMI class imbalance can otherwise cause the model to collapse "
                            "to always predicting the majority class N)")
    parser.add_argument("--class-balanced-max-oversample", type=float, default=50.0,
                       help="Max oversample multiplier for rare classes when --class-balanced is set (default: 50.0)")

    args = parser.parse_args()

    llm_models = [m.strip() for m in args.llm_models.split(",") if m.strip()]

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = fixed_seeds[:2]

    if args.epochs is not None:
        train_epochs = args.epochs
    else:
        train_epochs = 0 if args.mode == "inference" else 10

    if args.mode == "inference" and not args.checkpoint_path:
        raise ValueError("--checkpoint-path required for inference mode")

    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = f"./experiments/time_llm_ecg_classifier_{args.mode}_mitbih/"

    length_sets = get_length_sets(args.mode)
    torch_dtypes = [args.torch_dtype]
    print(f"🚀 Starting {args.mode} config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print("🗃️  Dataset: mitbih")
    print(f"🤖 LLM Models: {llm_models}")
    print(f"🎲 Seeds: {seeds}")
    print(f"📈 Epochs: {train_epochs}")
    if args.mode == "inference":
        print(f"📦 Checkpoint: {args.checkpoint_path}")

    config_count = 0
    for seed, llm_model_name, length_set, torch_dtype in product(seeds, llm_models, length_sets, torch_dtypes):
        llm_config = get_llm_config(llm_model_name)
        seq_len = length_set["sequence_length"]
        patch_len = length_set["patch_len"]

        folder_name = f"seed_{seed}_model_{llm_config['llm_model']}_dim_{llm_config['llm_dim']}_seq_{seq_len}_patch_{patch_len}_epochs_{train_epochs}"
        experiment_folder = os.path.join(base_output_dir, folder_name)
        dataset_folder = os.path.join(experiment_folder, "dataset_mitbih")
        log_folder = os.path.join(dataset_folder, "logs")

        os.makedirs(dataset_folder, exist_ok=True)
        os.makedirs(log_folder, exist_ok=True)

        config_content = generate_config_content(
            args.mode,
            seed,
            llm_config,
            length_set,
            train_epochs=train_epochs,
            checkpoint_path=args.checkpoint_path,
            torch_dtype=torch_dtype,
            include_duplicate_202=args.include_duplicate_202,
            class_balanced=args.class_balanced,
            class_balanced_max_oversample=args.class_balanced_max_oversample,
        )

        config_content = config_content.replace("LOGS_PLACEHOLDER", log_folder)

        config_path = os.path.join(dataset_folder, "config.gin")
        with open(config_path, "w") as f:
            f.write(config_content)

        print(f"Generated: {config_path}")
        config_count += 1

    print(f"✅ Generated {config_count} MIT-BIH ECG config files in {base_output_dir}")


if __name__ == "__main__":
    main()
