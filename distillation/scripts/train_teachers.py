#!/usr/bin/env python3
"""
Automated Teacher Training Script for Time-LLM Models
Trains BERT and Tiny BERT teacher models with configurable parameters.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_configs_path


class TeacherTrainer:
    """Class to handle automated training of teacher models."""
    
    def __init__(self, base_dir=None, output_dir=None, config_dir=None):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.base_dir = Path(base_dir)
        
        # Set config directory - use pipeline config dir if provided, otherwise root configs
        if config_dir:
            self.configs_dir = Path(config_dir)
        else:
            self.configs_dir = get_configs_path()
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set results directory
        if output_dir:
            self.results_dir = Path(output_dir)
        else:
            self.results_dir = self.base_dir / "results" / "teacher_models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define teacher model configurations
        self.teacher_models = {
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "model_comment": "time_llm_BERT_768_6_6_9_6"
            },
            "distilbert": {
                "llm_model": "DistilBERT", 
                "llm_layers": 6,
                "llm_dim": 768,
                "model_comment": "time_llm_distilBERT_768_6_6_9_6"
            },
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "model_comment": "time_llm_TinyBERT_312_6_6_9_6"
            }
        }
        
        # Base training configuration template - will be updated based on dataset
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
                "method": "time_llm",
                "num_workers": 1,
                "torch_dtype": "float32",
                "model_id": "teacher",
                "sequence_length": 6,
                "context_length": 6,
                "prediction_length": 9,
                "patch_len": 6,
                "stride": 8,
                "prediction_batch_size": 64,
                "train_batch_size": 32,
                "learning_rate": 0.001,
                "train_epochs": 20,
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
                "des": "teacher_training",
                "prompt_domain": 0,
                "timeenc": 0,
                "eval_metrics": ["rmse", "mae", "mape"],
                "seed": 238822
            }
        }

    def generate_config(self, model_name, dataset="584", epochs=20):
        """Generate gin config file for a specific teacher model."""
        if model_name not in self.teacher_models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(self.teacher_models.keys())}")
        
        # Determine the correct data paths based on dataset
        if dataset in ['540', '544', '552', '559', '563', '567', '570', '575', '584', '588', '591', '596']:
            # OhioT1DM dataset
            data_dir = "./data/ohiot1dm/raw_standardized"
            train_file = f"{data_dir}/{dataset}-ws-training.csv"
            test_file = f"{data_dir}/{dataset}-ws-testing.csv"
            prompt_file = f"{data_dir}/t1dm_prompt.txt"
        else:
            # D1NAMO dataset (001-007)
            data_dir = "./data/d1namo/raw_standardized"
            train_file = f"{data_dir}/{dataset}-ws-training.csv"
            test_file = f"{data_dir}/{dataset}-ws-testing.csv"
            prompt_file = f"{data_dir}/t1dm_prompt.txt"
        
        config = self.base_config.copy()
        model_config = self.teacher_models[model_name]
        
        # Update data paths
        config["data_settings"]["path_to_train_data"] = train_file
        config["data_settings"]["path_to_test_data"] = test_file
        config["data_settings"]["prompt_path"] = prompt_file
        
        # Update model-specific settings
        config["llm_settings"].update(model_config)
        config["llm_settings"]["train_epochs"] = epochs
        
        # Generate log directory path using output directory if provided
        if hasattr(self, 'results_dir') and str(self.results_dir) != str(self.base_dir / "results" / "teacher_models"):
            log_dir = f"{self.results_dir}/{model_name}_{dataset}_{epochs}epochs/logs"
        else:
            log_dir = f"./results/teacher_models/{model_name}_{dataset}_{epochs}epochs/logs"
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
        
        return config_content

    def save_config(self, config_content, model_name, dataset="584", epochs=20):
        """Save config content to a gin file."""
        config_filename = f"config_teacher_{model_name}_{dataset}_{epochs}epochs.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def train_model(self, model_name, dataset="584", epochs=20, dry_run=False):
        """Train a specific teacher model."""
        print(f"\n{'='*60}")
        print(f"Training Teacher Model: {model_name.upper()}")
        print(f"Dataset: {dataset}, Epochs: {epochs}")
        print(f"{'='*60}")
        
        # Generate and save config
        config_content = self.generate_config(model_name, dataset, epochs)
        config_path = self.save_config(config_content, model_name, dataset, epochs)
        
        print(f"✓ Config generated: {config_path}")
        
        if dry_run:
            print("DRY RUN: Would execute training command")
            return config_path
        
        # Execute training command
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
                print(f"✓ Successfully trained {model_name} teacher model")
                
                # Create training summary
                self.create_training_summary(model_name, dataset, epochs, config_path)
                
                return config_path
            else:
                print(f"✗ Training failed with return code: {result.returncode}")
                return None
        
        except Exception as e:
            print(f"✗ Error during training: {str(e)}")
            return None
        finally:
            os.chdir(original_dir)

    def create_training_summary(self, model_name, dataset, epochs, config_path):
        """Create a summary of the training run."""
        summary = {
            "model_name": model_name,
            "dataset": dataset,
            "epochs": epochs,
            "config_path": str(config_path),
            "model_config": self.teacher_models[model_name],
            "status": "completed"
        }
        
        summary_file = self.results_dir / f"{model_name}_{dataset}_{epochs}epochs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Training summary saved: {summary_file}")

    def train_all_teachers(self, dataset="584", epochs=20, dry_run=False):
        """Train all teacher models."""
        print(f"\n{'='*80}")
        print("AUTOMATED TEACHER MODEL TRAINING PIPELINE")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name in self.teacher_models.keys():
            try:
                config_path = self.train_model(model_name, dataset, epochs, dry_run)
                results[model_name] = {
                    "status": "success" if config_path else "failed",
                    "config_path": str(config_path) if config_path else None
                }
            except Exception as e:
                print(f"✗ Failed to train {model_name}: {str(e)}")
                results[model_name] = {"status": "error", "error": str(e)}
        
        # Save overall results
        results_file = self.results_dir / f"teacher_training_results_{dataset}_{epochs}epochs.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("TEACHER TRAINING COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return results

    def list_available_checkpoints(self):
        """List available teacher model checkpoints."""
        print("\nAvailable Teacher Model Checkpoints:")
        print("-" * 50)
        
        for model_dir in self.results_dir.glob("*"):
            if model_dir.is_dir():
                checkpoint_files = list(model_dir.glob("**/*.pth"))
                if checkpoint_files:
                    print(f"\n{model_dir.name}:")
                    for checkpoint in checkpoint_files:
                        print(f"  - {checkpoint}")


def main():
    parser = argparse.ArgumentParser(description="Train Time-LLM Teacher Models")
    parser.add_argument("--model", choices=["bert", "distilbert", "tinybert", "all"], 
                       default="all", help="Which model to train")
    parser.add_argument("--dataset", default="584", help="Dataset identifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't train")
    parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints")
    parser.add_argument("--output-dir", help="Output directory for trained models (default: results/teacher_models)")
    parser.add_argument("--config-dir", help="Directory for saving config files (default: configs)")
    
    args = parser.parse_args()
    
    trainer = TeacherTrainer(output_dir=args.output_dir, config_dir=args.config_dir)
    
    if args.list_checkpoints:
        trainer.list_available_checkpoints()
        return
    
    if args.model == "all":
        trainer.train_all_teachers(args.dataset, args.epochs, args.dry_run)
    else:
        trainer.train_model(args.model, args.dataset, args.epochs, args.dry_run)


if __name__ == "__main__":
    main()