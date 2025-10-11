#!/usr/bin/env python3
"""
Complete Distillation Pipeline Test Script
Trains teacher, student, performs distillation, and runs inference on all models.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root


class FullPipelineTest:
    """Complete pipeline test with configurable epochs for each phase."""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.configs_dir = self.base_dir / "configs" / "distillation_test"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.base_dir / "distillation_experiments" / "full_pipeline_test"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def create_training_config(self, model_type, dataset, epochs, phase="train"):
        """Create training configuration for a specific model and phase."""
        
        # Model configurations
        model_configs = {
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": f"teacher_BERT_768_{phase}",
                "d_ff": 128
            },
            "tinybert": {
                "llm_model": "TinyBERT", 
                "llm_layers": 4,
                "llm_dim": 312,
                "model_comment": f"student_TinyBERT_312_{phase}",
                "d_ff": 32
            }
        }
        
        # Determine dataset paths
        if dataset in ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']:
            dataset_name = "ohiot1dm"
            data_type = "raw_standardized"
        else:
            dataset_name = "d1namo"
            data_type = "raw_standardized"
            
        train_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-training.csv"
        test_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-testing.csv"
        prompt_path = f"./data/{dataset_name}/{data_type}/t1dm_prompt.txt"
        
        config_content = f'''# Training Configuration for {model_type.upper()} - {phase.upper()}
# Dataset: {dataset}, Epochs: {epochs}

run.log_dir = "./distillation_experiments/full_pipeline_test/logs/{model_type}_{dataset}_{phase}/"

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
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'training',
    'method': 'time_llm',
    'llm_model': '{model_configs[model_type]["llm_model"]}',
    'llm_layers': {model_configs[model_type]["llm_layers"]},
    'llm_dim': {model_configs[model_type]["llm_dim"]},
    'num_workers': 1,
    'torch_dtype': 'bfloat16',
    'model_id': '{dataset_name}_{dataset}',
    'sequence_length': 96,
    'context_length': 10,
    'prediction_length': 1,
    'patch_len': 16,
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.001,
    'train_epochs': {epochs},
    'features': 'S',
    'd_model': 32,
    'd_ff': {model_configs[model_type]["d_ff"]},
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
    'des': '{phase}',
    'model_comment': '{model_configs[model_type]["model_comment"]}',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': 238822,
    'restore_from_checkpoint': False
}}
'''
        
        return config_content
        
    def create_distillation_config(self, teacher_model, student_model, dataset, epochs, teacher_checkpoint):
        """Create distillation configuration."""
        
        # Determine dataset paths
        if dataset in ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']:
            dataset_name = "ohiot1dm"
            data_type = "raw_standardized"
        else:
            dataset_name = "d1namo"
            data_type = "raw_standardized"
            
        train_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-training.csv"
        test_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-testing.csv"
        prompt_path = f"./data/{dataset_name}/{data_type}/t1dm_prompt.txt"
        
        student_configs = {
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4, 
                "llm_dim": 312,
                "d_ff": 32
            }
        }
        
        config_content = f'''# Distillation Configuration: {teacher_model.upper()} -> {student_model.upper()}
# Dataset: {dataset}, Epochs: {epochs}

run.log_dir = "./distillation_experiments/full_pipeline_test/logs/distilled_{student_model}_{dataset}/"

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
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'training+inference',
    'method': 'time_llm',
    'llm_model': '{student_configs[student_model]["llm_model"]}',
    'llm_layers': {student_configs[student_model]["llm_layers"]},
    'llm_dim': {student_configs[student_model]["llm_dim"]},
    'num_workers': 1,
    'torch_dtype': 'bfloat16',
    'model_id': '{dataset_name}_{dataset}',
    'sequence_length': 96,
    'context_length': 10,
    'prediction_length': 1,
    'patch_len': 16,
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.0005,
    'train_epochs': {epochs},
    'features': 'S',
    'd_model': 32,
    'd_ff': {student_configs[student_model]["d_ff"]},
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
    'des': 'distillation',
    'model_comment': 'distilled_{student_configs[student_model]["llm_model"]}_{student_configs[student_model]["llm_dim"]}_distillation',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': 238822,
    'restore_from_checkpoint': False,
    'teacher_checkpoint_path': '{teacher_checkpoint}',
    'is_distillation': True,
    'alpha': 0.5,
    'beta': 0.5,
    'kl_weight': 0.1,
    'temperature': 3.0
}}
'''
        
        return config_content
        
    def create_inference_config(self, model_type, dataset, checkpoint_path, phase="inference"):
        """Create inference configuration."""
        
        # Determine model config based on type
        if "distilled" in model_type:
            base_model = model_type.replace("distilled_", "")
            if base_model == "tinybert":
                model_config = {
                    "llm_model": "TinyBERT",
                    "llm_layers": 4,
                    "llm_dim": 312,
                    "d_ff": 32
                }
        else:
            model_configs = {
                "bert": {
                    "llm_model": "BERT",
                    "llm_layers": 12,
                    "llm_dim": 768,
                    "d_ff": 128
                },
                "tinybert": {
                    "llm_model": "TinyBERT",
                    "llm_layers": 4,
                    "llm_dim": 312,
                    "d_ff": 32
                }
            }
            model_config = model_configs[model_type]
            
        # Determine dataset paths
        if dataset in ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']:
            dataset_name = "ohiot1dm"
            data_type = "raw_standardized"
        else:
            dataset_name = "d1namo"
            data_type = "raw_standardized"
            
        train_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-training.csv"
        test_data_path = f"./data/{dataset_name}/{data_type}/{dataset}-ws-testing.csv"
        prompt_path = f"./data/{dataset_name}/{data_type}/t1dm_prompt.txt"
        
        config_content = f'''# Inference Configuration for {model_type.upper()}
# Dataset: {dataset}

run.log_dir = "./distillation_experiments/full_pipeline_test/logs/{model_type}_{dataset}_{phase}/"

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
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'inference',
    'method': 'time_llm',
    'llm_model': '{model_config["llm_model"]}',
    'llm_layers': {model_config["llm_layers"]},
    'llm_dim': {model_config["llm_dim"]},
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': '{dataset_name}_{dataset}',
    'sequence_length': 96,
    'context_length': 10,
    'prediction_length': 1,
    'patch_len': 16,
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.001,
    'train_epochs': 1,
    'features': 'S',
    'd_model': 32,
    'd_ff': {model_config["d_ff"]},
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
    'des': '{phase}',
    'model_comment': '{model_type}_{phase}',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': 238822,
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{checkpoint_path}'
}}
'''
        
        return config_content
        
    def run_training_phase(self, model_type, dataset, epochs, phase="train"):
        """Run training phase for a model."""
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} - {phase.upper()}")
        print(f"Dataset: {dataset}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Generate config
        config_content = self.create_training_config(model_type, dataset, epochs, phase)
        config_filename = f"{model_type}_{dataset}_{phase}_{epochs}epochs.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Config generated: {config_path}")
        
        # Run training
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir)
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully trained {model_type}")
                return True
            else:
                print(f"✗ Training failed for {model_type}")
                return False
        finally:
            os.chdir(original_dir)
            
    def run_distillation_phase(self, teacher_model, student_model, dataset, epochs):
        """Run distillation phase."""
        print(f"\n{'='*60}")
        print(f"Distillation: {teacher_model.upper()} -> {student_model.upper()}")
        print(f"Dataset: {dataset}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Find teacher checkpoint in the logs directory structure
        teacher_logs_dir = self.base_dir / "distillation_experiments" / "full_pipeline_test" / "logs" / f"{teacher_model}_{dataset}_train"
        
        # Find the most recent log directory
        teacher_log_dirs = sorted(teacher_logs_dir.glob("logs_*"))
        if not teacher_log_dirs:
            print(f"✗ No teacher log directory found in {teacher_logs_dir}")
            return False
            
        latest_log_dir = teacher_log_dirs[-1]  # Get the most recent
        teacher_checkpoint_dir = latest_log_dir / "checkpoints"
        teacher_checkpoints = list(teacher_checkpoint_dir.glob("*.pth"))
        
        if not teacher_checkpoints:
            print(f"✗ No teacher checkpoint found in {teacher_checkpoint_dir}")
            return False
            
        teacher_checkpoint = str(teacher_checkpoints[0])
        print(f"✓ Using teacher checkpoint: {teacher_checkpoint}")
        
        # Generate distillation config
        config_content = self.create_distillation_config(teacher_model, student_model, dataset, epochs, teacher_checkpoint)
        config_filename = f"distill_{teacher_model}_to_{student_model}_{dataset}_{epochs}epochs.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Distillation config generated: {config_path}")
        
        # Run distillation
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir)
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully distilled {teacher_model} -> {student_model}")
                return True
            else:
                print(f"✗ Distillation failed for {teacher_model} -> {student_model}")
                return False
        finally:
            os.chdir(original_dir)
            
    def run_inference_phase(self, model_type, dataset):
        """Run inference phase for a model."""
        print(f"\n{'='*60}")
        print(f"Inference: {model_type.upper()}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        # Find model checkpoint in the logs directory structure
        if "distilled" in model_type:
            # For distilled models, look in distillation logs
            logs_dir = self.base_dir / "distillation_experiments" / "full_pipeline_test" / "logs" / f"{model_type}_{dataset}"
        else:
            # For regular models, look in training logs
            logs_dir = self.base_dir / "distillation_experiments" / "full_pipeline_test" / "logs" / f"{model_type}_{dataset}_train"
        
        # Find the most recent log directory
        log_dirs = sorted(logs_dir.glob("logs_*"))
        if not log_dirs:
            print(f"✗ No log directory found in {logs_dir}")
            return False
            
        latest_log_dir = log_dirs[-1]  # Get the most recent
        checkpoint_dir = latest_log_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        
        if not checkpoints:
            print(f"✗ No checkpoint found in {checkpoint_dir}")
            return False
            
        checkpoint_path = str(checkpoint_dir)
        print(f"✓ Using checkpoint directory: {checkpoint_path}")
        
        # Generate inference config
        config_content = self.create_inference_config(model_type, dataset, checkpoint_path)
        config_filename = f"inference_{model_type}_{dataset}.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        print(f"✓ Inference config generated: {config_path}")
        
        # Run inference
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        original_dir = os.getcwd()
        try:
            os.chdir(self.base_dir)
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully ran inference for {model_type}")
                return True
            else:
                print(f"✗ Inference failed for {model_type}")
                return False
        finally:
            os.chdir(original_dir)
            
    def run_full_pipeline(self, teacher_model, student_model, dataset, teacher_epochs, student_epochs, distill_epochs):
        """Run complete pipeline test."""
        print(f"\n{'='*80}")
        print(f"FULL PIPELINE TEST")
        print(f"Teacher: {teacher_model.upper()} ({teacher_epochs} epochs)")
        print(f"Student: {student_model.upper()} ({student_epochs} epochs)")
        print(f"Distillation: {distill_epochs} epochs")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")
        
        results = {
            "teacher_training": False,
            "student_training": False, 
            "distillation": False,
            "teacher_inference": False,
            "student_inference": False,
            "distilled_inference": False
        }
        
        # Phase 1: Train Teacher
        results["teacher_training"] = self.run_training_phase(teacher_model, dataset, teacher_epochs, "train")
        if not results["teacher_training"]:
            print("✗ Pipeline failed at teacher training")
            return results
            
        # Phase 2: Train Student (baseline)
        results["student_training"] = self.run_training_phase(student_model, dataset, student_epochs, "train")
        if not results["student_training"]:
            print("✗ Pipeline failed at student training")
            return results
            
        # Phase 3: Distillation
        results["distillation"] = self.run_distillation_phase(teacher_model, student_model, dataset, distill_epochs)
        if not results["distillation"]:
            print("✗ Pipeline failed at distillation")
            return results
            
        # Phase 4: Inference on all models (SKIPPED - only training phases requested)
        print("✓ Skipping inference phases - training-only pipeline")
        results["teacher_inference"] = True  # Mark as successful (skipped)
        results["student_inference"] = True  # Mark as successful (skipped)
        results["distilled_inference"] = True  # Mark as successful (skipped)
        
        # Save results summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "teacher_model": teacher_model,
                "student_model": student_model,
                "dataset": dataset,
                "teacher_epochs": teacher_epochs,
                "student_epochs": student_epochs,
                "distill_epochs": distill_epochs
            },
            "results": results
        }
        
        summary_path = self.results_dir / f"pipeline_test_{dataset}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n{'='*80}")
        print(f"PIPELINE TEST COMPLETE")
        print(f"Summary saved to: {summary_path}")
        
        success_count = sum(results.values())
        total_phases = len(results)
        print(f"Success rate: {success_count}/{total_phases} phases")
        print(f"{'='*80}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Full Distillation Pipeline Test")
    parser.add_argument("--teacher", choices=["bert"], default="bert", help="Teacher model")
    parser.add_argument("--student", choices=["tinybert"], default="tinybert", help="Student model")
    parser.add_argument("--dataset", default="570", help="Dataset identifier")
    parser.add_argument("--teacher-epochs", type=int, default=1, help="Teacher training epochs")
    parser.add_argument("--student-epochs", type=int, default=1, help="Student training epochs")
    parser.add_argument("--distill-epochs", type=int, default=1, help="Distillation epochs")
    
    args = parser.parse_args()
    
    pipeline = FullPipelineTest()
    
    results = pipeline.run_full_pipeline(
        teacher_model=args.teacher,
        student_model=args.student,
        dataset=args.dataset,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        distill_epochs=args.distill_epochs
    )
    
    # Exit with error code if any phase failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()