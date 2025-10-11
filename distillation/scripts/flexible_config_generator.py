#!/usr/bin/env python3
"""
Flexible Configuration Generator for Time-LLM Experiments
Provides full control over datasets, data types, patients, and models.
"""

import json
import os
import sys
import glob
from pathlib import Path
import argparse
from datetime import datetime

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root, get_data_path


class FlexibleConfigGenerator:
    """Generate configurations with full control over all parameters."""
    
    def __init__(self, base_dir=None, configs_dir=None):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        self.data_dir = get_data_path()
        if configs_dir:
            self.configs_dir = Path(configs_dir)
        else:
            self.configs_dir = self.base_dir / "distillation_experiments" / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Available models
        self.models = {
            # Teacher Models
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "method": "time_llm",
                "type": "teacher"
            },
            "distilbert": {
                "llm_model": "DistilBERT",
                "llm_layers": 6,
                "llm_dim": 768,
                "method": "time_llm",
                "type": "teacher"
            },
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "method": "time_llm",
                "type": "teacher"
            },
            "gpt2": {
                "llm_model": "GPT2",
                "llm_layers": 12,
                "llm_dim": 768,
                "method": "time_llm",
                "type": "teacher"
            },
            "minilm": {
                "llm_model": "MiniLM",
                "llm_layers": 6,
                "llm_dim": 384,
                "method": "time_llm",
                "type": "teacher"
            },
            "mobilebert": {
                "llm_model": "MobileBERT",
                "llm_layers": 24,
                "llm_dim": 512,
                "method": "time_llm",
                "type": "teacher"
            },
            "albert": {
                "llm_model": "ALBERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "method": "time_llm",
                "type": "teacher"
            },
            "bert_tiny": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "method": "time_llm",
                "type": "teacher"
            },
            "opt_125m": {
                "llm_model": "OPT-125M",
                "llm_layers": 12,
                "llm_dim": 768,
                "method": "time_llm",
                "type": "teacher"
            },
            "chronos": {
                "llm_model": "Chronos",
                "llm_layers": 4,
                "llm_dim": 512,
                "method": "time_llm",
                "type": "teacher"
            },
            # Student Models
            "tinybert_student": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "method": "student_llm",
                "type": "student"
            },
            "distilbert_student": {
                "llm_model": "DistilBERT",
                "llm_layers": 6,
                "llm_dim": 768,
                "method": "student_llm",
                "type": "student"
            },
            "bert_tiny_student": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "method": "student_llm",
                "type": "student"
            },
            "minilm_student": {
                "llm_model": "MiniLM",
                "llm_layers": 6,
                "llm_dim": 384,
                "method": "student_llm",
                "type": "student"
            },
            "mobilebert_student": {
                "llm_model": "MobileBERT",
                "llm_layers": 24,
                "llm_dim": 512,
                "method": "student_llm",
                "type": "student"
            }
        }
        
        # Base configuration template
        self.base_config_template = {
            "llm_settings": {
                "task_name": "long_term_forecast",
                "mode": "training+inference",
                "num_workers": 1,
                "torch_dtype": "float32",
                "model_id": "experiment",
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
                "des": "experiment",
                "prompt_domain": 0,
                "timeenc": 0,
                "eval_metrics": ["rmse", "mae", "mape"],
                "seed": 238822
            },
            "data_settings": {
                "input_features": ["target"],
                "labels": ["target"],
                "preprocessing_method": "min_max",
                "preprocess_input_features": False,
                "preprocess_label": False,
                "frequency": "5min",
                "percent": 100,
                "val_split": 0
            }
        }

    def discover_datasets(self):
        """Discover available datasets, data types, and patients."""
        datasets = {}
        
        # Check each dataset directory
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                datasets[dataset_name] = {
                    "data_types": {},
                    "path": str(dataset_dir)
                }
                
                # Check each data type within dataset
                for data_type_dir in dataset_dir.iterdir():
                    if data_type_dir.is_dir():
                        data_type = data_type_dir.name
                        datasets[dataset_name]["data_types"][data_type] = {
                            "patients": [],
                            "path": str(data_type_dir)
                        }
                        
                        # Find available patients
                        patients = set()
                        for file_path in data_type_dir.glob("*-ws-*.csv"):
                            patient_id = file_path.name.split('-')[0]
                            patients.add(patient_id)
                        
                        datasets[dataset_name]["data_types"][data_type]["patients"] = sorted(list(patients))
                        
                        # Check for prompt file
                        prompt_file = data_type_dir / "t1dm_prompt.txt"
                        datasets[dataset_name]["data_types"][data_type]["prompt_file"] = str(prompt_file) if prompt_file.exists() else None
        
        return datasets

    def generate_config(self, dataset, data_type, patient_id, model_name, 
                       train_epochs=20, learning_rate=0.001, batch_size=32,
                       sequence_length=6, prediction_length=9, mode="training+inference",
                       custom_params=None):
        """Generate a configuration for specific parameters."""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.models.keys())}")
        
        # Build data paths
        data_path = self.data_dir / dataset / data_type
        train_file = data_path / f"{patient_id}-ws-training.csv"
        test_file = data_path / f"{patient_id}-ws-testing.csv"
        prompt_file = data_path / "t1dm_prompt.txt"
        
        # Verify files exist
        if not train_file.exists():
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Testing file not found: {test_file}")
        
        # Create configuration
        config = self.base_config_template.copy()
        
        # Update data settings
        config["data_settings"].update({
            "path_to_train_data": str(train_file),
            "path_to_test_data": str(test_file),
            "prompt_path": str(prompt_file) if prompt_file.exists() else None
        })
        
        # Update model settings
        model_config = self.models[model_name]
        config["llm_settings"].update(model_config)
        
        # Update training parameters
        config["llm_settings"].update({
            "train_epochs": train_epochs,
            "learning_rate": learning_rate,
            "train_batch_size": batch_size,
            "sequence_length": sequence_length,
            "prediction_length": prediction_length,
            "mode": mode,
            "model_comment": f"{model_config['llm_model']}_{dataset}_{data_type}_{patient_id}_{train_epochs}epochs"
        })
        
        # Apply custom parameters if provided
        if custom_params:
            if "llm_settings" in custom_params:
                config["llm_settings"].update(custom_params["llm_settings"])
            if "data_settings" in custom_params:
                config["data_settings"].update(custom_params["data_settings"])
        
        # Generate log directory following experiments structure
        experiment_id = f"{model_name}_{dataset}_{data_type}_{patient_id}_{train_epochs}epochs"
        model_config = self.models[model_name]
        experiment_folder = f"seed_{config['llm_settings']['seed']}_model_{model_config['llm_model']}_dim_{model_config['llm_dim']}_seq_{config['llm_settings']['sequence_length']}_context_{config['llm_settings']['context_length']}_pred_{config['llm_settings']['prediction_length']}_patch_{config['llm_settings']['patch_len']}_epochs_{train_epochs}"
        log_dir = f"./distillation_experiments/{dataset}_distillation/{experiment_folder}/patient_{patient_id}/logs"
        
        return config, experiment_id, log_dir

    def save_config(self, config, experiment_id, log_dir, format_type="gin"):
        """Save configuration in specified format."""
        
        if format_type == "gin":
            return self._save_gin_config(config, experiment_id, log_dir)
        elif format_type == "json":
            return self._save_json_config(config, experiment_id, log_dir)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Use 'gin' or 'json'")

    def _save_gin_config(self, config, experiment_id, log_dir):
        """Save as gin configuration file."""
        config_filename = f"config_{experiment_id}.gin"
        config_path = self.configs_dir / config_filename
        
        config_content = f'run.log_dir = "{log_dir}"\n\n'
        
        # Data settings
        config_content += "run.data_settings = {\n"
        for key, value in config["data_settings"].items():
            if value is None:
                continue
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n\n"
        
        # LLM settings
        config_content += "run.llm_settings = {\n"
        for key, value in config["llm_settings"].items():
            if value is None:
                continue
            if isinstance(value, str):
                config_content += f"    '{key}': '{value}',\n"
            elif isinstance(value, list):
                config_content += f"    '{key}': {value},\n"
            else:
                config_content += f"    '{key}': {value},\n"
        config_content += "}\n"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def _save_json_config(self, config, experiment_id, log_dir):
        """Save as JSON configuration file."""
        config_filename = f"config_{experiment_id}.json"
        config_path = self.configs_dir / config_filename
        
        full_config = {
            "log_dir": log_dir,
            "experiment_id": experiment_id,
            **config
        }
        
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        return config_path

    def create_experiment_batch(self, experiments_spec, batch_name=None):
        """Create multiple configurations from experiment specification."""
        
        if batch_name is None:
            batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        batch_dir = self.configs_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        created_configs = []
        
        for exp in experiments_spec:
            try:
                config, experiment_id, log_dir = self.generate_config(**exp)
                config_path = batch_dir / f"config_{experiment_id}.gin"
                
                # Save gin config in batch directory
                config_content = f'run.log_dir = "{log_dir}"\n\n'
                
                # Data settings
                config_content += "run.data_settings = {\n"
                for key, value in config["data_settings"].items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        config_content += f"    '{key}': '{value}',\n"
                    else:
                        config_content += f"    '{key}': {value},\n"
                config_content += "}\n\n"
                
                # LLM settings
                config_content += "run.llm_settings = {\n"
                for key, value in config["llm_settings"].items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        config_content += f"    '{key}': '{value}',\n"
                    elif isinstance(value, list):
                        config_content += f"    '{key}': {value},\n"
                    else:
                        config_content += f"    '{key}': {value},\n"
                config_content += "}\n"
                
                with open(config_path, 'w') as f:
                    f.write(config_content)
                
                created_configs.append({
                    "experiment_id": experiment_id,
                    "config_path": str(config_path),
                    "config": config
                })
                
            except Exception as e:
                print(f"Failed to create config for {exp}: {str(e)}")
        
        # Save batch summary
        batch_summary = {
            "batch_name": batch_name,
            "created_at": datetime.now().isoformat(),
            "total_experiments": len(created_configs),
            "experiments": created_configs
        }
        
        summary_file = batch_dir / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        print(f"Created {len(created_configs)} configurations in batch: {batch_name}")
        print(f"Batch directory: {batch_dir}")
        
        return batch_summary

    def list_available_data(self):
        """List all available datasets, data types, and patients."""
        datasets = self.discover_datasets()
        
        print("Available Data:")
        print("=" * 50)
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\n📁 Dataset: {dataset_name}")
            print(f"   Path: {dataset_info['path']}")
            
            for data_type, type_info in dataset_info['data_types'].items():
                print(f"   📂 Data Type: {data_type}")
                print(f"      Patients: {', '.join(type_info['patients'])}")
                print(f"      Prompt: {'✓' if type_info['prompt_file'] else '✗'}")
        
        return datasets

    def list_available_models(self):
        """List all available models."""
        print("\nAvailable Models:")
        print("=" * 50)
        
        teachers = {k: v for k, v in self.models.items() if v["type"] == "teacher"}
        students = {k: v for k, v in self.models.items() if v["type"] == "student"}
        
        print("\n🎓 Teacher Models:")
        for name, config in teachers.items():
            print(f"   {name}: {config['llm_model']} ({config['llm_layers']} layers, {config['llm_dim']} dim)")
        
        print("\n🎒 Student Models:")
        for name, config in students.items():
            print(f"   {name}: {config['llm_model']} ({config['llm_layers']} layers, {config['llm_dim']} dim)")


def main():
    parser = argparse.ArgumentParser(description="Flexible Configuration Generator for Time-LLM")
    
    # Discovery commands
    parser.add_argument("--list-data", action="store_true", help="List available datasets and patients")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    # Single config generation
    parser.add_argument("--dataset", help="Dataset name (e.g., d1namo, ohiot1dm)")
    parser.add_argument("--data-type", help="Data type (e.g., raw_standardized, missing_periodic_standardized)")
    parser.add_argument("--patient", help="Patient ID (e.g., 001, 002)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--mode", default="training+inference", help="Mode (training+inference, training, inference)")
    parser.add_argument("--format", default="gin", choices=["gin", "json"], help="Config format")
    
    # Batch generation
    parser.add_argument("--batch-file", help="JSON file with batch experiment specifications")
    parser.add_argument("--batch-name", help="Name for the batch")
    
    args = parser.parse_args()
    
    generator = FlexibleConfigGenerator()
    
    if args.list_data:
        generator.list_available_data()
        return
    
    if args.list_models:
        generator.list_available_models()
        return
    
    if args.batch_file:
        with open(args.batch_file, 'r') as f:
            experiments_spec = json.load(f)
        generator.create_experiment_batch(experiments_spec, args.batch_name)
        return
    
    if args.dataset and args.data_type and args.patient and args.model:
        try:
            config, experiment_id, log_dir = generator.generate_config(
                dataset=args.dataset,
                data_type=args.data_type, 
                patient_id=args.patient,
                model_name=args.model,
                train_epochs=args.epochs,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                mode=args.mode
            )
            
            config_path = generator.save_config(config, experiment_id, log_dir, args.format)
            print(f"✓ Configuration created: {config_path}")
            print(f"✓ Experiment ID: {experiment_id}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    else:
        print("Error: Missing required parameters for single config generation")
        parser.print_help()


if __name__ == "__main__":
    main()