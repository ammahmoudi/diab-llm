#!/usr/bin/env python3
"""Generate Time-LLM ECG distillation configs for MIT-BIH.

This mirrors the Time-LLM config-generator style, but targets the ECG
classification distillation path.
"""

from __future__ import annotations

import argparse
import json
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
                            alpha=0.5, beta=0.5, temperature=2.0,
                            fair_teacher=False, fair_teacher_feature="sex",
                            student_calibration=False, student_calibration_feature="sex",
                            teacher_calibration=False, teacher_calibration_feature="sex",
                            teacher_calibration_offsets=None,
                            fairness_constraint=False, fairness_constraint_feature="sex",
                            fairness_constraint_target_classes=None,
                            fairness_constraint_epsilon=0.05, fairness_dual_lr=0.01, fairness_dual_init=1.0,
                            feature_alignment=False, feature_alignment_feature="sex",
                            feature_alignment_weight=0.0, feature_alignment_divergence="coral",
                            kd_replay=False, kd_replay_feature="sex", kd_replay_minority_group=None,
                            kd_replay_target_classes=None, kd_replay_factor=4.0,
                            adv_erasure=False, adv_erasure_feature="sex", adv_erasure_lambda=1.0,
                            multi_teacher=False, multi_teacher_feature="sex", multi_teacher_group0=None,
                            second_teacher_checkpoint_path=None):
    log_folder_placeholder = "LOGS_PLACEHOLDER"
    mode_str = "training+inference" if mode != "inference" else "inference"
    batch_sizes = get_model_batch_sizes(student_config["llm_model"])
    restore_flag = mode == "inference"

    fair_teacher_line = (
        f"\n     'fair_teacher_sampling': True,"
        f"\n     'fair_teacher_feature': '{fair_teacher_feature}',"
        if fair_teacher else ""
    )

    # K1, O1, K3, K4, O3, T2: extra llm_settings keys, only emitted when enabled.
    fixes_lines = ""
    if teacher_calibration:
        fixes_lines += (
            f"\n     'teacher_calibration_enabled': True,"
            f"\n     'teacher_calibration_feature': '{teacher_calibration_feature}',"
            f"\n     'teacher_calibration_offsets': {repr(teacher_calibration_offsets or {})},"
        )
    if fairness_constraint:
        fixes_lines += (
            f"\n     'fairness_constraint_enabled': True,"
            f"\n     'fairness_constraint_feature': '{fairness_constraint_feature}',"
            f"\n     'fairness_constraint_target_classes': {repr(fairness_constraint_target_classes or [1, 2, 3, 4])},"
            f"\n     'fairness_constraint_epsilon': {fairness_constraint_epsilon},"
            f"\n     'fairness_dual_lr': {fairness_dual_lr},"
            f"\n     'fairness_dual_init': {fairness_dual_init},"
        )
    if feature_alignment:
        fixes_lines += (
            f"\n     'feature_alignment_enabled': True,"
            f"\n     'feature_alignment_feature': '{feature_alignment_feature}',"
            f"\n     'feature_alignment_weight': {feature_alignment_weight},"
            f"\n     'feature_alignment_divergence': '{feature_alignment_divergence}',"
        )
    if kd_replay:
        fixes_lines += (
            f"\n     'kd_replay_enabled': True,"
            f"\n     'kd_replay_feature': '{kd_replay_feature}',"
            f"\n     'kd_replay_minority_group': '{kd_replay_minority_group}',"
            f"\n     'kd_replay_target_classes': {repr(kd_replay_target_classes or [1, 2, 3, 4])},"
            f"\n     'kd_replay_factor': {kd_replay_factor},"
        )
    if adv_erasure:
        fixes_lines += (
            f"\n     'adv_erasure_enabled': True,"
            f"\n     'adv_erasure_feature': '{adv_erasure_feature}',"
            f"\n     'adv_erasure_lambda': {adv_erasure_lambda},"
        )
    if multi_teacher:
        fixes_lines += (
            f"\n     'multi_teacher_enabled': True,"
            f"\n     'multi_teacher_feature': '{multi_teacher_feature}',"
            f"\n     'multi_teacher_group0': '{multi_teacher_group0}',"
            f"\n     'second_teacher_checkpoint_path': '{second_teacher_checkpoint_path}',"
        )

    config_content = f'''# Parameters for run:
# ==============================================================================
run.data_settings = \\
    {{'dataset_dir': './data/mit-bih-arrhythmia',
     'metadata_csv': './data/mit-bih-arrhythmia/metadata_records.csv',
     'demographics_csv': './data/mit-bih-arrhythmia/demographics_records.csv',
     'beat_index_csv': './data/mit-bih-arrhythmia/beat_index.csv',
     'include_duplicate_202': {str(include_duplicate_202)},{fair_teacher_line}}}

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
     'student_calibration_enabled': {str(student_calibration)},
     'student_calibration_feature': '{student_calibration_feature}',
     'task_name': 'ecg_classification',
     'teacher_checkpoint_path': '{teacher_checkpoint_path}',
     'teacher_model': '{teacher_config["llm_model"]}',
     'timeenc': 0,
     'torch_dtype': '{torch_dtype}',
     'train_batch_size': {batch_sizes["train_batch_size"]},
     'train_epochs': {train_epochs},{fixes_lines}}}
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
    parser.add_argument("--fair-teacher", action="store_true",
                       help="Enable ECG fair teacher sampling (T1 analogue): upweight rare group/class combinations")
    parser.add_argument("--fair-teacher-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"],
                       help="Demographic feature for fair teacher sampling (default: sex)")
    parser.add_argument("--student-calibration", action="store_true",
                       help="Enable O2-style learned per-group logit calibration on the distilled student")
    parser.add_argument("--student-calibration-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"],
                       help="Demographic feature for student calibration (default: sex)")
    parser.add_argument("--teacher-calibration", action="store_true",
                       help="Enable K1-style calibrated soft labels (per-group teacher logit offsets)")
    parser.add_argument("--teacher-calibration-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--teacher-calibration-offsets", default=None,
                       help="JSON dict of group -> per-class logit offsets, e.g. '{\"M\": [0,0,0,0,0], \"F\": [0,0.5,0,0,0]}'")
    parser.add_argument("--fairness-constraint", action="store_true",
                       help="Enable O1-style fairness constraint via projected dual ascent")
    parser.add_argument("--fairness-constraint-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--fairness-constraint-target-classes", default="1,2,3,4",
                       help="Comma-separated AAMI class ids treated as fairness-critical (default: S,V,F,Q)")
    parser.add_argument("--fairness-constraint-epsilon", type=float, default=0.05)
    parser.add_argument("--fairness-dual-lr", type=float, default=0.01)
    parser.add_argument("--fairness-dual-init", type=float, default=1.0)
    parser.add_argument("--feature-alignment", action="store_true",
                       help="Enable K3-style fairness-aware feature alignment on student hidden states")
    parser.add_argument("--feature-alignment-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--feature-alignment-weight", type=float, default=1.0)
    parser.add_argument("--feature-alignment-divergence", default="coral", choices=["coral", "mmd"])
    parser.add_argument("--kd-replay", action="store_true",
                       help="Enable K4-style selective KD replay (upweight minority-group target-class samples)")
    parser.add_argument("--kd-replay-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--kd-replay-minority-group", default=None,
                       help="Group value treated as minority for KD replay, e.g. 'F'")
    parser.add_argument("--kd-replay-target-classes", default="1,2,3,4",
                       help="Comma-separated AAMI class ids to upweight (default: S,V,F,Q)")
    parser.add_argument("--kd-replay-factor", type=float, default=4.0)
    parser.add_argument("--adv-erasure", action="store_true",
                       help="Enable O3-style adversarial group erasure (gradient-reversal discriminator)")
    parser.add_argument("--adv-erasure-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--adv-erasure-lambda", type=float, default=1.0)
    parser.add_argument("--multi-teacher", action="store_true",
                       help="Enable T2-style multi-teacher KD (route samples to a group-specialized second teacher)")
    parser.add_argument("--multi-teacher-feature", default="sex",
                       choices=["sex", "age_group", "paced_group", "difficulty_group"])
    parser.add_argument("--multi-teacher-group0", default=None,
                       help="Group value served by the second teacher, e.g. 'F'")
    parser.add_argument("--second-teacher-checkpoint-path", default=None,
                       help="Path to the group-0 specialized second teacher checkpoint (required with --multi-teacher)")
    args = parser.parse_args()

    if args.multi_teacher and not args.second_teacher_checkpoint_path:
        parser.error("--multi-teacher requires --second-teacher-checkpoint-path")

    teacher_calibration_offsets = json.loads(args.teacher_calibration_offsets) if args.teacher_calibration_offsets else None
    fairness_constraint_target_classes = [int(c) for c in args.fairness_constraint_target_classes.split(',') if c.strip()]
    kd_replay_target_classes = [int(c) for c in args.kd_replay_target_classes.split(',') if c.strip()]

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
            fair_teacher=args.fair_teacher,
            fair_teacher_feature=args.fair_teacher_feature,
            student_calibration=args.student_calibration,
            student_calibration_feature=args.student_calibration_feature,
            teacher_calibration=args.teacher_calibration,
            teacher_calibration_feature=args.teacher_calibration_feature,
            teacher_calibration_offsets=teacher_calibration_offsets,
            fairness_constraint=args.fairness_constraint,
            fairness_constraint_feature=args.fairness_constraint_feature,
            fairness_constraint_target_classes=fairness_constraint_target_classes,
            fairness_constraint_epsilon=args.fairness_constraint_epsilon,
            fairness_dual_lr=args.fairness_dual_lr,
            fairness_dual_init=args.fairness_dual_init,
            feature_alignment=args.feature_alignment,
            feature_alignment_feature=args.feature_alignment_feature,
            feature_alignment_weight=args.feature_alignment_weight,
            feature_alignment_divergence=args.feature_alignment_divergence,
            kd_replay=args.kd_replay,
            kd_replay_feature=args.kd_replay_feature,
            kd_replay_minority_group=args.kd_replay_minority_group,
            kd_replay_target_classes=kd_replay_target_classes,
            kd_replay_factor=args.kd_replay_factor,
            adv_erasure=args.adv_erasure,
            adv_erasure_feature=args.adv_erasure_feature,
            adv_erasure_lambda=args.adv_erasure_lambda,
            multi_teacher=args.multi_teacher,
            multi_teacher_feature=args.multi_teacher_feature,
            multi_teacher_group0=args.multi_teacher_group0,
            second_teacher_checkpoint_path=args.second_teacher_checkpoint_path,
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
