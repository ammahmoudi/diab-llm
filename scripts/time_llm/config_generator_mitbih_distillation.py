#!/usr/bin/env python3
"""Generate Time-LLM ECG distillation configs for MIT-BIH.

This mirrors the Time-LLM config-generator style, but targets the ECG
classification distillation path.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.seeds import fixed_seeds
from time_llm.config_generator import get_llm_config, get_model_batch_sizes


def get_length_sets(mode):
    return [
        {"sequence_length": 256, "context_length": 256, "prediction_length": 0, "patch_len": 16},
        {"sequence_length": 300, "context_length": 300, "prediction_length": 0, "patch_len": 20},
    ]


def generate_config_content(mode, seed, teacher_config, student_config, length_set,
                            train_epochs=10, teacher_checkpoint_path=None,
                            torch_dtype="float32", include_duplicate_202=False,
                            alpha=0.5, beta=0.5, temperature=2.0):
    log_folder_placeholder = "LOGS_PLACEHOLDER"
    mode_str = "training+inference" if mode != "inference" else "inference"
    batch_sizes = get_model_batch_sizes(student_config["llm_model"])
    restore_flag = mode == "inference"

    config_content = f'''# Parameters for run:
# ==============================================================================
run.data_settings = \\
    {{'dataset_dir': './data/mit-bih-arrhythmia',
     'metadata_csv': './data/mit-bih-arrhythmia/metadata_records.csv',
     'demographics_csv': './data/mit-bih-arrhythmia/demographics_records.csv',
     'beat_index_csv': './data/mit-bih-arrhythmia/beat_index.csv',
     'include_duplicate_202': {str(include_duplicate_202)}}}

run.llm_settings = \\
    {{'activation': 'gelu',
     'c_out': 5,
     'context_length': {length_set["context_length"]},
     'd_ff': 128,
     'd_layers': 1,
     'd_model': 64,
     'dec_in': 1,
     'des': 'mitbih_distillation',
     'distillation_alpha': {alpha},
     'distillation_beta': {beta},
     'distillation_temperature': {temperature},
     'dropout': 0.1,
     'e_layers': 2,
     'embed': 'timeF',
     'enc_in': 1,
     'eval_metrics': ['accuracy', 'macro_f1', 'weighted_f1'],
     'factor': 1,
     'learning_rate': 0.0001,
     'llm_dim': {student_config["llm_dim"]},
     'llm_layers': {student_config["llm_layers"]},
     'llm_model': '{student_config["llm_model"]}',
     'lradj': 'COS',
     'method': 'distillation_ecg_classifier',
     'mode': '{mode_str}',
     'model_comment': 'time_llm_ecg_distill_teacher_{teacher_config["llm_model"]}_student_{student_config["llm_model"]}_{length_set["sequence_length"]}_{length_set["patch_len"]}',
     'model_id': 'mitbih_distillation',
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
     'restore_checkpoint_path': '',
     'seed': {seed},
     'sequence_length': {length_set["sequence_length"]},
     'stride': 8,
     'task_name': 'ecg_classification',
     'teacher_checkpoint_path': '{teacher_checkpoint_path}',
     'teacher_model': '{teacher_config["llm_model"]}',
     'timeenc': 0,
     'torch_dtype': '{torch_dtype}',
     'train_batch_size': {batch_sizes["train_batch_size"]},
     'train_epochs': {train_epochs}}}
run.log_dir = \\
    '{log_folder_placeholder}'
'''
    return config_content


def main():
    parser = argparse.ArgumentParser(description="MIT-BIH Time-LLM Distillation Configuration Generator")
    parser.add_argument("--mode", required=True, choices=["train", "inference", "train_inference"], help="Operation mode")
    parser.add_argument("--teacher-model", default="BERT", help="Teacher LLM model name")
    parser.add_argument("--student-models", default="TinyBERT,BERT-tiny", help="Comma-separated student model names")
    parser.add_argument("--teacher-checkpoint-path", required=True, help="Path to trained teacher checkpoint")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--epochs", type=int, default=None, help="Distillation epochs")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--include-duplicate-202", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    teacher_config = get_llm_config(args.teacher_model)
    student_models = [m.strip() for m in args.student_models.split(',') if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',')] if args.seeds else fixed_seeds[:2]
    train_epochs = args.epochs if args.epochs is not None else (0 if args.mode == 'inference' else 10)

    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = f"./experiments/time_llm_ecg_classifier_distillation_{args.mode}_mitbih/"

    length_sets = get_length_sets(args.mode)
    config_count = 0

    print(f"🚀 Starting MIT-BIH distillation config generation...")
    print(f"📁 Output directory: {base_output_dir}")
    print(f"👨‍🏫 Teacher model: {teacher_config['llm_model']}")
    print(f"👨‍🎓 Student models: {student_models}")

    for seed, student_model_name, length_set in product(seeds, student_models, length_sets):
        student_config = get_llm_config(student_model_name)
        seq_len = length_set['sequence_length']
        patch_len = length_set['patch_len']
        folder_name = (
            f"seed_{seed}_teacher_{teacher_config['llm_model']}_{teacher_config['llm_dim']}"
            f"_student_{student_config['llm_model']}_{student_config['llm_dim']}"
            f"_seq_{seq_len}_patch_{patch_len}_epochs_{train_epochs}"
        )
        experiment_folder = os.path.join(base_output_dir, folder_name)
        dataset_folder = os.path.join(experiment_folder, 'dataset_mitbih')
        log_folder = os.path.join(dataset_folder, 'logs')
        os.makedirs(log_folder, exist_ok=True)

        config_content = generate_config_content(
            args.mode,
            seed,
            teacher_config,
            student_config,
            length_set,
            train_epochs=train_epochs,
            teacher_checkpoint_path=args.teacher_checkpoint_path,
            torch_dtype=args.torch_dtype,
            include_duplicate_202=args.include_duplicate_202,
            alpha=args.alpha,
            beta=args.beta,
            temperature=args.temperature,
        )
        config_content = config_content.replace('LOGS_PLACEHOLDER', log_folder)
        config_path = os.path.join(dataset_folder, 'config.gin')
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Generated: {config_path}")
        config_count += 1

    print(f"✅ Generated {config_count} MIT-BIH ECG distillation config files in {base_output_dir}")


if __name__ == '__main__':
    main()
