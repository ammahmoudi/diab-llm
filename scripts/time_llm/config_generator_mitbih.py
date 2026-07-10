#!/usr/bin/env python3
"""Generate Time-LLM ECG classification configs for MIT-BIH.

This keeps the MIT-BIH path close to the repository's existing config-generator
pattern while avoiding disruptive changes to the large unified generator.
"""

from __future__ import annotations

import argparse
import os
from itertools import product
from pathlib import Path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.seeds import fixed_seeds
from time_llm.config_generator import get_llm_config, get_model_batch_sizes


def get_length_sets():
    return [
        {"sequence_length": 256, "context_length": 256, "prediction_length": 0, "patch_len": 16},
        {"sequence_length": 300, "context_length": 300, "prediction_length": 0, "patch_len": 20},
    ]


def generate_config_content(mode, seed, llm_config, length_set, train_epochs=10, torch_dtype="float32", include_duplicate_202=False):
    log_folder_placeholder = "LOGS_PLACEHOLDER"
    mode_str = "inference" if mode == "inference" else "training+inference" if mode == "train_inference" else "training"
    batch_sizes = get_model_batch_sizes(llm_config["llm_model"])
    return f'''# Parameters for run:
# ==============================================================================
run.data_settings = \\
    {{'dataset_dir': './data/mit-bih-arrhythmia',
     'metadata_csv': './data/mit-bih-arrhythmia/metadata_records.csv',
     'demographics_csv': './data/mit-bih-arrhythmia/demographics_records.csv',
     'beat_index_csv': './data/mit-bih-arrhythmia/beat_index.csv',
     'include_duplicate_202': {str(include_duplicate_202)}}}

run.llm_settings = \\
    {{'task_name': 'ecg_classification',
     'mode': '{mode_str}',
     'method': 'time_llm_ecg_classifier',
     'restore_from_checkpoint': False,
     'llm_model': '{llm_config["llm_model"]}',
     'llm_layers': {llm_config["llm_layers"]},
     'llm_dim': {llm_config["llm_dim"]},
     'num_workers': 0,
     'torch_dtype': '{torch_dtype}',
     'model_id': 'mitbih',
     'sequence_length': {length_set["sequence_length"]},
     'context_length': {length_set["context_length"]},
     'prediction_length': 0,
     'patch_len': {length_set["patch_len"]},
     'stride': 8,
     'prediction_batch_size': {batch_sizes['prediction_batch_size']},
     'train_batch_size': {batch_sizes['train_batch_size']},
     'learning_rate': 0.0001,
     'train_epochs': {train_epochs},
     'd_model': 64,
     'd_ff': 128,
     'factor': 1,
     'enc_in': 1,
     'dec_in': 1,
     'c_out': 5,
     'e_layers': 2,
     'd_layers': 1,
     'n_heads': 8,
     'dropout': 0.1,
     'moving_avg': 25,
     'activation': 'gelu',
     'embed': 'timeF',
     'patience': 10,
     'lradj': 'COS',
     'des': 'mitbih',
     'model_comment': 'time_llm_ecg_{llm_config["llm_model"]}_{llm_config["llm_dim"]}_{length_set["sequence_length"]}_{length_set["patch_len"]}',
     'prompt_domain': 0,
     'timeenc': 0,
     'pooling': 'mean',
     'num_classes': 5,
     'freeze_llm': False,
     'eval_metrics': ['accuracy', 'macro_f1', 'weighted_f1'],
     'seed': {seed}}}
run.log_dir = \\
    '{log_folder_placeholder}'
'''


def main():
    parser = argparse.ArgumentParser(description="MIT-BIH Time-LLM ECG config generator")
    parser.add_argument("--mode", required=True, choices=["train", "inference", "train_inference"])
    parser.add_argument("--llm_models", default="TinyBERT,BERT-tiny")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--include-duplicate-202", action="store_true")
    args = parser.parse_args()

    llm_models = [m.strip() for m in args.llm_models.split(',')]
    seeds = [int(s.strip()) for s in args.seeds.split(',')] if args.seeds else fixed_seeds[:2]
    train_epochs = args.epochs if args.epochs is not None else (0 if args.mode == 'inference' else 10)

    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = f"./experiments/time_llm_ecg_classifier_{args.mode}_mitbih/"

    length_sets = get_length_sets()
    config_count = 0
    for seed, llm_model_name, length_set in product(seeds, llm_models, length_sets):
        llm_config = get_llm_config(llm_model_name)
        folder_name = (
            f"seed_{seed}_model_{llm_config['llm_model']}_dim_{llm_config['llm_dim']}"
            f"_seq_{length_set['sequence_length']}_patch_{length_set['patch_len']}_epochs_{train_epochs}"
        )
        experiment_folder = os.path.join(base_output_dir, folder_name)
        config_folder = os.path.join(experiment_folder, 'dataset_mitbih')
        log_folder = os.path.join(config_folder, 'logs')
        os.makedirs(log_folder, exist_ok=True)

        content = generate_config_content(
            args.mode,
            seed,
            llm_config,
            length_set,
            train_epochs=train_epochs,
            torch_dtype=args.torch_dtype,
            include_duplicate_202=args.include_duplicate_202,
        )
        content = content.replace('LOGS_PLACEHOLDER', log_folder)
        config_path = os.path.join(config_folder, 'config.gin')
        with open(config_path, 'w') as f:
            f.write(content)
        print(f"Generated: {config_path}")
        config_count += 1

    print(f"✅ Generated {config_count} MIT-BIH ECG config files in {base_output_dir}")


if __name__ == '__main__':
    main()
