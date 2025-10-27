#!/usr/bin/env python3
"""
Automated Student Distillation Script for Time-LLM Models
Performs knowledge distillation from teacher models to student models.
"""

import os
import sys
import subprocess
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_configs_path


class DistillationTrainer:
    """Class to handle automated distillation training."""
    
    def __init__(self, base_dir=None, distill_epochs=1, 
                 teacher_checkpoint_dir=None, student_config_dir=None, 
                 output_dir=None, config_output_dir=None, pipeline_dir=None,
                 dataset_name="ohiot1dm", seed=238822, lr=0.001, batch_size=32,
                 alpha=0.5, beta=0.5):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        
        # Model name mappings from HuggingFace names to internal config names
        self.model_name_mapping = {
            # BERT variants
            "bert": "bert",
            "bert-base-uncased": "bert",
            "bert-base-cased": "bert",
            "bert-large-uncased": "bert-large",
            "bert-large-cased": "bert-large",
            # DistilBERT variants
            "distilbert": "distilbert",
            "distilbert-base-uncased": "distilbert",
            "distilbert-base-cased": "distilbert",
            # TinyBERT and small models (consistent with other scripts)
            "tinybert": "tinybert",
            "huawei-noah/TinyBERT_General_4L_312D": "tinybert",
            "prajjwal1/bert-tiny": "bert-tiny",  # Fixed: use dash not underscore
            "prajjwal1/bert-mini": "bert-mini",  # Fixed: use dash not underscore
            "prajjwal1/bert-small": "bert-small",  # Fixed: use dash not underscore
            "prajjwal1/bert-medium": "bert-medium",  # Fixed: use dash not underscore
            # Additional models from Time-LLM
            "gpt2": "gpt2",
            "openai-community/gpt2": "gpt2",
            "llama": "llama",
            "huggyllama/llama-7b": "llama",
            "minilm": "minilm",
            "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large": "minilm",
            "mobilebert": "mobilebert",
            "google/mobilebert-uncased": "mobilebert",
            "albert": "albert",
            "albert/albert-base-v2": "albert",
            "opt-125m": "opt-125m",
            "facebook/opt-125m": "opt-125m",
        }
        
        # Set directory paths based on parameters
        if config_output_dir:
            self.configs_dir = Path(config_output_dir)
        else:
            self.configs_dir = self.base_dir / "configs" / "distillation"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        if output_dir:
            self.results_dir = Path(output_dir)
        else:
            self.results_dir = self.base_dir / "distillation_experiments" / "distilled_models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        if teacher_checkpoint_dir:
            self.teacher_results_dir = Path(teacher_checkpoint_dir)
        else:
            self.teacher_results_dir = self.base_dir / "results" / "teacher_models"
            
        self.student_config_dir = Path(student_config_dir) if student_config_dir else None
        self.pipeline_dir = pipeline_dir
        self.dataset_name = dataset_name
        self.data_path = f"./data/{dataset_name}"  # Auto-detect data path
        self.seed = seed
        
        # Store epoch parameters
        self.distill_epochs = distill_epochs
        
        # Define student model configurations (extended to support all Time-LLM models)
        self.student_models = {
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "model_comment": "student_TinyBERT_312_6_6_9_6"
            },
            "distilbert": {
                "llm_model": "DistilBERT",
                "llm_layers": 6, 
                "llm_dim": 768,
                "model_comment": "student_DistilBERT_768_6_6_9_6"
            },
            "bert-tiny": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "model_comment": "student_BERT_tiny_128_6_6_9_6"
            },
            "bert-mini": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 256,
                "model_comment": "student_BERT_256_6_6_9_6"
            },
            "bert-small": {
                "llm_model": "BERT",
                "llm_layers": 4,
                "llm_dim": 512,
                "model_comment": "student_BERT_512_6_6_9_6"
            },
            "bert-medium": {
                "llm_model": "BERT",
                "llm_layers": 8,
                "llm_dim": 512,
                "model_comment": "student_BERT_512_6_6_9_6"
            },
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_BERT_768_6_6_9_6"
            },
            "gpt2": {
                "llm_model": "GPT2",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_GPT2_768_6_6_9_6"
            },
            "minilm": {
                "llm_model": "MiniLM",
                "llm_layers": 6,
                "llm_dim": 384,
                "model_comment": "student_MiniLM_384_6_6_9_6"
            },
            "mobilebert": {
                "llm_model": "MobileBERT",
                "llm_layers": 24,
                "llm_dim": 512,
                "model_comment": "student_MobileBERT_512_6_6_9_6"
            },
            "albert": {
                "llm_model": "ALBERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_ALBERT_768_6_6_9_6"
            },
            "opt-125m": {
                "llm_model": "OPT-125M",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "student_OPT_125M_768_6_6_9_6"
            }
        }
        
        # Distillation hyperparameters
        self.distillation_params = {
            "alpha": self.alpha,        # Weight for ground truth loss
            "beta": self.beta,         # Weight for teacher output loss  
            "train_epochs": self.distill_epochs,   # Use provided distillation epochs
            "learning_rate": self.lr  # Use provided learning rate
        }
        
        # Base distillation configuration template - paths will be set dynamically
        self.base_config = {
            "data_settings": {
                "path_to_train_data": "",  # Will be set dynamically
                "path_to_test_data": "",   # Will be set dynamically
                "input_features": ["target"],
                "labels": ["target"],
                "prompt_path": "",  # Will be set dynamically
                "preprocessing_method": "min_max",
                "preprocess_input_features": False,
                "preprocess_label": False,
                "frequency": "5min",
                "percent": 100,
                "val_split": 0
            },
            "llm_settings": {
                "task_name": "long_term_forecast",
                "mode": "training+inference",
                "method": "distillation",
                "num_workers": 1,
                "torch_dtype": "float32",
                "model_id": "distilled_student",
                "sequence_length": 6,
                "context_length": 6,
                "prediction_length": 9,
                "patch_len": 6,
                "stride": 8,
                "prediction_batch_size": 64,
                "train_batch_size": 32,
                "features": "S",
                "d_model": 32,
                "d_ff": 32,
                "factor": 1,
                "enc_in": 1,
                "dec_in": 1,
                "c_out": 1,
                "e_layers": 2,
                "d_layers": 1,
                "n_heads": 8,
                "dropout": 0.1,
                "moving_avg": 25,
                "activation": "gelu",
                "embed": "timeF",
                "patience": 10,
                "lradj": "COS",
                "des": "distillation",
                "prompt_domain": 0,
                "timeenc": 0,
                "eval_metrics": ["rmse", "mae", "mape"],
                "seed": 238822,
                "restore_from_checkpoint": False,
                "train_epochs": self.distill_epochs,
                "teacher_checkpoint_path": ""  # Will be set dynamically
            }
        }

    def find_teacher_checkpoint(self, teacher_model, dataset="584"):
        """Find the checkpoint file for a trained teacher model."""
        # First, look in new distillation_experiments structure
        distil_exp_dir = self.base_dir / "distillation_experiments"
        
        # Determine dataset type for path construction
        if dataset in ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']:
            dataset_name = "ohiot1dm_distillation"
        else:
            dataset_name = "d1namo_distillation"
        
        # Look for teacher model in distillation_experiments
        teacher_model_upper = teacher_model.upper()
        if teacher_model_upper == "TINYBERT":
            teacher_model_upper = "TinyBERT"
        elif teacher_model_upper == "DISTILBERT":
            teacher_model_upper = "DistilBERT"
        
        exp_pattern = f"seed_*_model_{teacher_model_upper}_*"
        distil_dataset_dir = distil_exp_dir / dataset_name
        teacher_dirs = []
        
        if distil_dataset_dir.exists():
            teacher_dirs = list(distil_dataset_dir.glob(exp_pattern))
            # Filter for the specific patient
            teacher_dirs = [d for d in teacher_dirs if (d / f"patient_{dataset}").exists()]
        
        # Fallback: look in old teacher results directory
        if not teacher_dirs:
            teacher_pattern = f"{teacher_model}_{dataset}_*epochs"
            teacher_dirs = list(self.teacher_results_dir.glob(teacher_pattern))
        
        # Fallback: look in experiment_configs_distil directory
        if not teacher_dirs:
            exp_pattern = f"./experiment_configs_distil/seed_*_model_{teacher_model.upper()}_*"
            teacher_dirs = list(self.base_dir.glob(exp_pattern))
        
        if not teacher_dirs:
            raise FileNotFoundError(f"No trained teacher model found for {teacher_model}")
        
        # Use the most recent directory
        teacher_dir = sorted(teacher_dirs)[-1]
        
        # Look for checkpoint files in new structure
        patient_dir = teacher_dir / f"patient_{dataset}"
        if patient_dir.exists():
            # New distillation_experiments structure
            checkpoint_patterns = [
                "logs/*/checkpoints/*.pth",
                "logs/*/checkpoints/checkpoint.pth",
                "logs/*/*.pth", 
                "**/*.pth"
            ]
        else:
            # Old structure
            checkpoint_patterns = ["*.pth", "logs/*.pth", "**/*.pth"]
        
        search_dir = patient_dir if patient_dir.exists() else teacher_dir
        
        for pattern in checkpoint_patterns:
            checkpoints = list(search_dir.glob(pattern))
            if checkpoints:
                return sorted(checkpoints)[-1]  # Return most recent checkpoint
        
        raise FileNotFoundError(f"No checkpoint found in {search_dir}")

    def normalize_model_name(self, model_name):
        """Normalize HuggingFace model names to internal config names."""
        if model_name in self.model_name_mapping:
            return self.model_name_mapping[model_name]
        else:
            # If not in mapping, try to extract base model name
            model_lower = model_name.lower()
            if "bert-tiny" in model_lower or "tiny" in model_lower:
                return "bert_tiny"
            elif "bert-mini" in model_lower or "mini" in model_lower:
                return "bert_mini"
            elif "bert-small" in model_lower or "small" in model_lower:
                return "bert_small"
            elif "bert-medium" in model_lower or "medium" in model_lower:
                return "bert_medium"
            elif "bert-large" in model_lower:
                return "bert-large"
            elif "bert" in model_lower:
                return "bert"
            elif "distilbert" in model_lower:
                return "distilbert"
            elif "tinybert" in model_lower:
                return "tinybert"
            else:
                raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.model_name_mapping.keys())}")

    def sanitize_filename(self, name):
        """Sanitize model names for safe file/directory usage."""
        return name.replace('/', '_').replace('\\', '_')

    def generate_distillation_config(self, teacher_model, student_model, dataset="584"):
        """Generate gin config file for distillation."""
        if student_model not in self.student_models:
            raise ValueError(f"Student model {student_model} not supported. Choose from: {list(self.student_models.keys())}")
        
        # Find teacher checkpoint
        teacher_checkpoint = self.find_teacher_checkpoint(teacher_model, dataset)
        print(f"Found teacher checkpoint: {teacher_checkpoint}")
        
        # Build data paths using flexible parameters
        # Check if using all_patients combined data
        if dataset == "all_patients":
            data_dir = f"{self.data_path}/all_patients_combined"
            train_file = f"{data_dir}/all_patients_training.csv"
            test_file = f"{data_dir}/all_patients_testing.csv"
            prompt_file = f"{self.data_path}/raw_standardized/t1dm_prompt.txt"
        else:
            data_dir = f"{self.data_path}/raw_standardized"
            train_file = f"{data_dir}/{dataset}-ws-training.csv"
            test_file = f"{data_dir}/{dataset}-ws-testing.csv"
            prompt_file = f"{data_dir}/t1dm_prompt.txt"
        
        config = self.base_config.copy()
        student_config = self.student_models[student_model]
        
        # Update data paths
        config["data_settings"]["path_to_train_data"] = train_file
        config["data_settings"]["path_to_test_data"] = test_file
        config["data_settings"]["prompt_path"] = prompt_file
        
        # Update student model settings
        config["llm_settings"].update(student_config)
        config["llm_settings"].update(self.distillation_params)
        config["llm_settings"]["seed"] = self.seed
        config["llm_settings"]["train_batch_size"] = self.batch_size
        
        # Add teacher checkpoint path
        config["llm_settings"]["teacher_checkpoint_path"] = str(teacher_checkpoint)
        config["llm_settings"]["teacher_model"] = teacher_model.upper()
        
        # Generate log directory path - use pipeline directory if available
        if self.pipeline_dir:
            log_dir = os.path.join(self.pipeline_dir, "phase_3_distillation", f"{teacher_model}_to_{student_model}_{dataset}", "logs")
        else:
            log_dir = str(self.results_dir / f"{teacher_model}_to_{student_model}_{dataset}" / "logs")
        config_content = f'run.log_dir = "{log_dir}"\n'
        
        # Convert data_settings to gin format
        config_content += "run.data_settings = {\n"
        for key, value in config["data_settings"].items():
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n\n"
        
        # Convert llm_settings to gin format
        config_content += "run.llm_settings = {\n"
        for key, value in config["llm_settings"].items():
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            elif isinstance(value, list):
                config_content += f"    '{key}': {value},\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n"
        
        return config_content, teacher_checkpoint

    def save_distillation_config(self, config_content, teacher_model, student_model, dataset="584"):
        """Save distillation config to a gin file."""
        # Sanitize model names for filename
        safe_teacher_name = self.sanitize_filename(teacher_model)
        safe_student_name = self.sanitize_filename(student_model)
        config_filename = f"config_distill_{safe_teacher_name}_to_{safe_student_name}_{dataset}.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def distill_model(self, teacher_model, student_model, dataset="584", dry_run=False):
        """Perform distillation from teacher to student model."""
        # Normalize model names for internal processing
        normalized_teacher = self.normalize_model_name(teacher_model)
        normalized_student = self.normalize_model_name(student_model)
        
        print(f"\n{'='*60}")
        print(f"Knowledge Distillation")
        print(f"Teacher: {teacher_model.upper()} -> Student: {student_model.upper()}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        try:
            # Generate and save config using normalized names
            config_content, teacher_checkpoint = self.generate_distillation_config(normalized_teacher, normalized_student, dataset)
            config_path = self.save_distillation_config(config_content, teacher_model, student_model, dataset)
            
            print(f"✓ Config generated: {config_path}")
            print(f"✓ Using teacher checkpoint: {teacher_checkpoint}")
            
            if dry_run:
                print("DRY RUN: Would execute distillation command")
                return config_path
            
            # Execute distillation command using main.py
            cmd = [
                "python", "main.py", 
                "--config_path", str(config_path),
                "--log_level", "INFO"
            ]
            
            print(f"Executing: {' '.join(cmd)}")
            
            # Change to base directory for execution
            original_dir = os.getcwd()
            try:
                os.chdir(self.base_dir)
                
                result = subprocess.run(cmd, capture_output=False, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Successfully distilled {teacher_model} -> {student_model}")
                    
                    # Find and copy the distilled model to pipeline directory
                    self.copy_distilled_model(normalized_teacher, normalized_student, dataset)
                    
                    # Create distillation summary
                    self.create_distillation_summary(normalized_teacher, normalized_student, dataset, config_path, teacher_checkpoint)
                    
                    return config_path
                else:
                    print(f"✗ Distillation failed with return code: {result.returncode}")
                    return None
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            print(f"✗ Error during distillation: {str(e)}")
            return None

    def copy_distilled_model(self, teacher_model, student_model, dataset):
        """Copy the distilled model from default location to pipeline directory."""
        import shutil
        import glob
        
        # Look for the distilled model in common locations, including pipeline runs
        search_patterns = [
            self.base_dir / "distillation_experiments" / "pipeline_runs" / "pipeline_*" / "phase_3_distillation" / "*" / "logs" / "logs_*" / "student_distilled.pth",
            self.base_dir / "distillation_experiments" / "distillation_runs" / "distill_*" / "student_distilled.pth",
            self.base_dir / "student_distilled.pth",
            self.base_dir / "distillation_experiments" / "distilled_models" / "*" / "student_distilled.pth"
        ]
        
        distilled_model_path = None
        for pattern in search_patterns:
            matches = glob.glob(str(pattern))
            if matches:
                # Get the most recent file
                distilled_model_path = max(matches, key=os.path.getctime)
                break
                
        if distilled_model_path and os.path.exists(distilled_model_path):
            sanitized_teacher = self.sanitize_filename(teacher_model)
            sanitized_student = self.sanitize_filename(student_model)
            target_path = self.results_dir / f"{sanitized_teacher}_to_{sanitized_student}_{dataset}_distilled.pth"
            shutil.copy2(distilled_model_path, target_path)
            print(f"✓ Distilled model copied to: {target_path}")
        else:
            print("⚠️ Distilled model not found in expected locations")



    def create_distillation_summary(self, teacher_model, student_model, dataset, config_path, teacher_checkpoint):
        """Create a comprehensive summary of the distillation run."""
        # Find the latest model directory and results
        model_dir = None
        performance_metrics = None
        training_info = None
        
        # Look for model directory pattern: {teacher}_to_{student}_{dataset}
        model_pattern = f"{teacher_model}_to_{student_model}_{dataset}"
        for item in self.results_dir.iterdir():
            if item.is_dir() and item.name.startswith(model_pattern):
                model_dir = item
                break
        
        if model_dir:
            # Look for logs directory
            logs_dirs = list(model_dir.glob("logs/logs_*"))
            if logs_dirs:
                latest_log_dir = max(logs_dirs, key=lambda x: x.name)
                
                # Look for predictions and targets files (from DistillationWrapper)
                predictions_file = latest_log_dir / "predictions.pkl"
                targets_file = latest_log_dir / "targets.pkl"
                
                if predictions_file.exists() and targets_file.exists():
                    try:
                        import pickle
                        import numpy as np
                        
                        with open(predictions_file, 'rb') as f:
                            predictions = pickle.load(f)
                        with open(targets_file, 'rb') as f:
                            targets = pickle.load(f)
                        
                        # Calculate metrics
                        y_true = np.array(targets).flatten()
                        y_pred = np.array(predictions).flatten()
                        
                        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                        mae = np.mean(np.abs(y_true - y_pred))
                        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                        
                        performance_metrics = {
                            "rmse": float(rmse),
                            "mae": float(mae),
                            "mape": float(mape)
                        }
                        
                    except Exception as e:
                        print(f"Warning: Could not calculate metrics from prediction files: {e}")
                
                # Extract training information
                training_info = {
                    "model_directory": str(model_dir),
                    "logs_directory": str(latest_log_dir),
                    "student_checkpoint": str(latest_log_dir / "student_distilled.pth") if (latest_log_dir / "student_distilled.pth").exists() else None
                }
        
        summary = {
            "teacher_model": teacher_model,
            "student_model": student_model,
            "dataset": dataset,
            "config_path": str(config_path),
            "teacher_checkpoint": str(teacher_checkpoint),
            "student_config": self.student_models[student_model],
            "distillation_params": self.distillation_params,
            "training_info": training_info,
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Sanitize model names for filename
        safe_teacher_name = self.sanitize_filename(teacher_model)
        safe_student_name = self.sanitize_filename(student_model)
        summary_file = self.results_dir / f"{safe_teacher_name}_to_{safe_student_name}_{dataset}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also save with standardized name for pipeline CSV logging
        standard_summary_file = self.results_dir / "distillation_summary.json"
        with open(standard_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Enhanced distillation summary saved: {summary_file}")
        print(f"✓ Standardized summary saved: {standard_summary_file}")
        if performance_metrics:
            print(f"  📊 Distilled Performance: RMSE={performance_metrics.get('rmse', 'N/A'):.3f}, MAE={performance_metrics.get('mae', 'N/A'):.3f}, MAPE={performance_metrics.get('mape', 'N/A'):.3f}%")

    def distill_all_combinations(self, dataset="584", dry_run=False):
        """Perform distillation for all teacher-student combinations."""
        print(f"\n{'='*80}")
        print("AUTOMATED KNOWLEDGE DISTILLATION PIPELINE")
        print(f"{'='*80}")
        
        # Define reasonable teacher-student pairs
        distillation_pairs = [
            ("bert", "tinybert"),      # BERT -> TinyBERT
            ("bert", "distilbert"),    # BERT -> DistilBERT
            ("distilbert", "tinybert"), # DistilBERT -> TinyBERT 
            ("distilbert", "bert_tiny") # DistilBERT -> BERT-tiny
        ]
        
        results = {}
        
        for teacher, student in distillation_pairs:
            pair_name = f"{teacher}_to_{student}"
            try:
                config_path = self.distill_model(teacher, student, dataset, dry_run)
                results[pair_name] = {
                    "status": "success" if config_path else "failed",
                    "config_path": str(config_path) if config_path else None,
                    "teacher": teacher,
                    "student": student
                }
            except Exception as e:
                print(f"✗ Failed distillation {teacher} -> {student}: {str(e)}")
                results[pair_name] = {
                    "status": "error", 
                    "error": str(e),
                    "teacher": teacher,
                    "student": student
                }
        
        # Save overall results
        results_file = self.results_dir / f"distillation_results_{dataset}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("KNOWLEDGE DISTILLATION COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return results

    def list_available_models(self):
        """List available teacher models and student options."""
        print("\nAvailable Teacher Models:")
        print("-" * 30)
        for teacher_dir in self.teacher_results_dir.glob("*"):
            if teacher_dir.is_dir():
                print(f"  - {teacher_dir.name}")
        
        print("\nAvailable Student Models:")
        print("-" * 30)
        for student in self.student_models.keys():
            print(f"  - {student}")

    def list_distilled_models(self):
        """List available distilled model checkpoints.""" 
        print("\nDistilled Model Checkpoints:")
        print("-" * 40)
        
        for model_dir in self.results_dir.glob("*"):
            if model_dir.is_dir():
                checkpoint_files = list(model_dir.glob("**/*.pth"))
                if checkpoint_files:
                    print(f"\n{model_dir.name}:")
                    for checkpoint in checkpoint_files:
                        print(f"  - {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Perform Knowledge Distillation for Time-LLM Models")
    parser.add_argument("--teacher", choices=["bert", "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", 
                                             "distilbert", "distilbert-base-uncased", "distilbert-base-cased", 
                                             "tinybert", "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "prajjwal1/bert-medium"], 
                       help="Teacher model to distill from")
    parser.add_argument("--student", choices=["bert", "bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased", 
                                             "distilbert", "distilbert-base-uncased", "distilbert-base-cased", 
                                             "tinybert", "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small", "prajjwal1/bert-medium"],
                       help="Student model to distill to")
    parser.add_argument("--patients", default="584", help="Patient IDs (comma-separated or single)")
    parser.add_argument("--all-patients", action="store_true", help="Train on ALL patients combined data (overrides --patients)")
    parser.add_argument("--dataset", default="ohiot1dm", help="Dataset name (ohiot1dm, d1namo)")
    parser.add_argument("--seed", type=int, default=238822, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for ground truth loss")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for teacher output loss")
    parser.add_argument("--all", action="store_true", help="Run all distillation combinations")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't train")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-distilled", action="store_true", help="List distilled model checkpoints")
    parser.add_argument("--distill-epochs", type=int, default=1, help="Number of epochs for distillation training")
    parser.add_argument("--teacher-checkpoint-dir", help="Directory containing teacher checkpoints")
    parser.add_argument("--student-config-dir", help="Directory containing student configs")
    parser.add_argument("--output-dir", help="Output directory for distillation results")
    parser.add_argument("--config-output-dir", help="Directory for saving distillation configs")
    parser.add_argument("--pipeline-dir", help="Pipeline directory for organized output structure")
    # NOTE: teacher-epochs and student-epochs removed - this script only does distillation!
    # Use train_teachers.py and flexible_experiment_runner.py for training
    
    args = parser.parse_args()
    
    distiller = DistillationTrainer(
        distill_epochs=args.distill_epochs,
        teacher_checkpoint_dir=args.teacher_checkpoint_dir,
        student_config_dir=args.student_config_dir,
        output_dir=args.output_dir,
        config_output_dir=args.config_output_dir,
        pipeline_dir=args.pipeline_dir,
        dataset_name=args.dataset,
        seed=args.seed,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta
    )
    
    if args.list_models:
        distiller.list_available_models()
        return
        
    if args.list_distilled:
        distiller.list_distilled_models()
        return
    
    # If --all-patients flag is set, use "all_patients" as the patient ID
    if args.all_patients:
        print("🌍 Distilling on ALL PATIENTS COMBINED")
        patient_id = "all_patients"
    else:
        patient_id = args.patients
    
    if args.all:
        results = distiller.distill_all_combinations(patient_id, args.dry_run)
        if results:
            # Check if any distillation succeeded
            success_count = sum(1 for result in results.values() if result.get("status") == "success")
            if success_count > 0:
                print(f"\n✅ Knowledge distillation completed successfully! ({success_count}/{len(results)} succeeded)")
                return 0
            else:
                print(f"\n❌ All knowledge distillation attempts failed!")
                return 1
        else:
            print(f"\n❌ Knowledge distillation failed!")
            return 1
    elif args.teacher and args.student:
        config_path = distiller.distill_model(args.teacher, args.student, patient_id, args.dry_run)
        if config_path:
            print(f"\n✅ Knowledge distillation completed successfully!")
            print(f"📁 Config: {config_path}")
            return 0
        else:
            print(f"\n❌ Knowledge distillation failed!")
            return 1
    else:
        print("Error: Specify --teacher and --student, or use --all for all combinations")
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())