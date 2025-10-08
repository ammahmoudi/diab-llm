#!/usr/bin/env python3
"""
Automated Inference Script for Time-LLM Models
Runs inference on teacher models, normally-trained students, and distilled students.
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


class InferenceRunner:
    """Class to handle automated inference on trained models."""
    
    def __init__(self, base_dir="/home/amma/LLM-TIME"):
        self.base_dir = Path(base_dir)
        self.configs_dir = self.base_dir / "configs" / "inference"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.base_dir / "results" / "inference"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories for different model types
        self.teacher_results_dir = self.base_dir / "results" / "teacher_models"
        self.distilled_results_dir = self.base_dir / "results" / "distilled_models"
        
        # Model configurations for inference
        self.model_configs = {
            "bert": {
                "llm_model": "BERT",
                "llm_layers": 12,
                "llm_dim": 768,
                "method": "time_llm"
            },
            "distilbert": {
                "llm_model": "DistilBERT", 
                "llm_layers": 6,
                "llm_dim": 768,
                "method": "time_llm"
            },
            "tinybert": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "method": "time_llm"
            },
            "tinybert_student": {
                "llm_model": "TinyBERT",
                "llm_layers": 4,
                "llm_dim": 312,
                "method": "student_llm"
            },
            "distilbert_student": {
                "llm_model": "DistilBERT",
                "llm_layers": 6,
                "llm_dim": 768,
                "method": "student_llm"
            },
            "bert_tiny_student": {
                "llm_model": "BERT-tiny",
                "llm_layers": 2,
                "llm_dim": 128,
                "method": "student_llm"
            }
        }
        
        # Base inference configuration template - paths will be set dynamically
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
                "mode": "inference",
                "num_workers": 1,
                "torch_dtype": "float32",
                "model_id": "inference",
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
                "des": "inference",
                "prompt_domain": 0,
                "timeenc": 0,
                "eval_metrics": ["rmse", "mae", "mape"],
                "seed": 238822,
                "restore_from_checkpoint": True
            }
        }

    def find_model_checkpoint(self, model_type, model_name, dataset="584"):
        """Find checkpoint file for a specific model."""
        checkpoints = []
        
        if model_type == "teacher":
            # Look in teacher results directory
            pattern = f"{model_name}_{dataset}_*epochs"
            model_dirs = list(self.teacher_results_dir.glob(pattern))
            
        elif model_type == "distilled":
            # Look in distilled results directory  
            pattern = f"*_to_{model_name}_{dataset}"
            model_dirs = list(self.distilled_results_dir.glob(pattern))
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        if not model_dirs:
            raise FileNotFoundError(f"No {model_type} model found for {model_name}")
        
        # Look for checkpoint files in all matching directories
        for model_dir in model_dirs:
            for pattern in ["*.pth", "logs/*.pth", "**/*.pth"]:
                checkpoints.extend(list(model_dir.glob(pattern)))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for {model_type} {model_name}")
        
        # Return the most recent checkpoint
        return sorted(checkpoints)[-1]

    def generate_inference_config(self, model_type, model_name, dataset="584"):
        """Generate gin config file for inference."""
        
        # Determine the config key based on model type and name
        if model_type == "teacher":
            if model_name in self.model_configs:
                config_key = model_name
            else:
                raise ValueError(f"Teacher model {model_name} not supported")
        elif model_type == "distilled":
            # For distilled models, use student version of the config
            config_key = f"{model_name}_student"
            if config_key not in self.model_configs:
                raise ValueError(f"Distilled model {model_name} not supported")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Find model checkpoint
        checkpoint_path = self.find_model_checkpoint(model_type, model_name, dataset)
        print(f"Found checkpoint: {checkpoint_path}")
        
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
        model_config = self.model_configs[config_key]
        
        # Update data paths
        config["data_settings"]["path_to_train_data"] = train_file
        config["data_settings"]["path_to_test_data"] = test_file
        config["data_settings"]["prompt_path"] = prompt_file
        
        # Update model-specific settings
        config["llm_settings"].update(model_config)
        config["llm_settings"]["restore_checkpoint_path"] = str(checkpoint_path)
        
        # Generate log directory path
        log_dir = f"./results/inference/{model_type}_{model_name}_{dataset}/logs"
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
        
        return config_content, checkpoint_path

    def save_inference_config(self, config_content, model_type, model_name, dataset="584"):
        """Save inference config to a gin file."""
        config_filename = f"config_inference_{model_type}_{model_name}_{dataset}.gin"
        config_path = self.configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        return config_path

    def run_inference(self, model_type, model_name, dataset="584", dry_run=False):
        """Run inference on a specific model."""
        print(f"\n{'='*60}")
        print(f"Running Inference")
        print(f"Type: {model_type.upper()}, Model: {model_name.upper()}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        
        try:
            # Generate and save config
            config_content, checkpoint_path = self.generate_inference_config(model_type, model_name, dataset)
            config_path = self.save_inference_config(config_content, model_type, model_name, dataset)
            
            print(f"✓ Config generated: {config_path}")
            print(f"✓ Using checkpoint: {checkpoint_path}")
            
            if dry_run:
                print("DRY RUN: Would execute inference command")
                return config_path, None
            
            # Execute inference command
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
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✓ Successfully completed inference for {model_type} {model_name}")
                    
                    # Extract metrics from output
                    metrics = self.extract_metrics_from_output(result.stdout)
                    
                    # Create inference summary
                    summary = self.create_inference_summary(model_type, model_name, dataset, config_path, checkpoint_path, metrics)
                    
                    return config_path, summary
                else:
                    print(f"✗ Inference failed with return code: {result.returncode}")
                    print("STDERR:", result.stderr)
                    return None, None
            finally:
                os.chdir(original_dir)
                
        except Exception as e:
            print(f"✗ Error during inference: {str(e)}")
            return None, None

    def extract_metrics_from_output(self, output):
        """Extract evaluation metrics from command output."""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "RMSE:" in line or "MAE:" in line or "MAPE:" in line:
                try:
                    # Parse lines like "RMSE: 0.1234"
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        metric_name = parts[0].strip().lower()
                        metric_value = float(parts[1].strip())
                        metrics[metric_name] = metric_value
                except:
                    continue
        
        return metrics

    def create_inference_summary(self, model_type, model_name, dataset, config_path, checkpoint_path, metrics):
        """Create a summary of the inference run."""
        summary = {
            "model_type": model_type,
            "model_name": model_name,
            "dataset": dataset,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        summary_file = self.results_dir / f"inference_{model_type}_{model_name}_{dataset}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Inference summary saved: {summary_file}")
        return summary

    def run_all_inference(self, dataset="584", dry_run=False):
        """Run inference on all available models."""
        print(f"\n{'='*80}")
        print("AUTOMATED INFERENCE PIPELINE")
        print(f"{'='*80}")
        
        results = []
        
        # Run inference on teacher models
        teacher_models = ["bert", "distilbert", "tinybert"]
        for model_name in teacher_models:
            try:
                config_path, summary = self.run_inference("teacher", model_name, dataset, dry_run)
                if summary:
                    results.append(summary)
            except FileNotFoundError as e:
                print(f"⚠️  Skipping teacher {model_name}: {str(e)}")
            except Exception as e:
                print(f"✗ Error with teacher {model_name}: {str(e)}")
        
        # Run inference on distilled models
        distilled_models = ["tinybert", "distilbert", "bert_tiny"]
        for model_name in distilled_models:
            try:
                config_path, summary = self.run_inference("distilled", model_name, dataset, dry_run)
                if summary:
                    results.append(summary)
            except FileNotFoundError as e:
                print(f"⚠️  Skipping distilled {model_name}: {str(e)}")
            except Exception as e:
                print(f"✗ Error with distilled {model_name}: {str(e)}")
        
        # Save comprehensive results
        results_file = self.results_dir / f"all_inference_results_{dataset}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison report
        self.create_comparison_report(results, dataset)
        
        print(f"\n{'='*80}")
        print("INFERENCE PIPELINE COMPLETE")
        print(f"Results saved to: {results_file}")
        print(f"{'='*80}")
        
        return results

    def create_comparison_report(self, results, dataset):
        """Create a comparison report of all models."""
        if not results:
            print("No results to compare")
            return
        
        # Convert to DataFrame for easy comparison
        comparison_data = []
        for result in results:
            row = {
                "Model_Type": result["model_type"],
                "Model_Name": result["model_name"],
                "Dataset": result["dataset"]
            }
            # Add metrics
            if result.get("metrics"):
                row.update(result["metrics"])
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save as CSV
        report_file = self.results_dir / f"model_comparison_{dataset}.csv"
        df.to_csv(report_file, index=False)
        
        print(f"✓ Comparison report saved: {report_file}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("No metrics available for comparison")

    def list_available_models(self):
        """List all available models for inference."""
        print("\nAvailable Models for Inference:")
        print("=" * 50)
        
        print("\nTeacher Models:")
        print("-" * 20)
        for teacher_dir in self.teacher_results_dir.glob("*"):
            if teacher_dir.is_dir():
                checkpoints = list(teacher_dir.glob("**/*.pth"))
                status = "✓" if checkpoints else "✗"
                print(f"  {status} {teacher_dir.name}")
        
        print("\nDistilled Models:")
        print("-" * 20)
        for distilled_dir in self.distilled_results_dir.glob("*"):
            if distilled_dir.is_dir():
                checkpoints = list(distilled_dir.glob("**/*.pth"))
                status = "✓" if checkpoints else "✗"
                print(f"  {status} {distilled_dir.name}")

    def list_inference_results(self):
        """List previous inference results."""
        print("\nPrevious Inference Results:")
        print("-" * 40)
        
        for result_file in self.results_dir.glob("inference_*.json"):
            print(f"  - {result_file.name}")
        
        for result_file in self.results_dir.glob("model_comparison_*.csv"):
            print(f"  - {result_file.name} (comparison report)")


def main():
    parser = argparse.ArgumentParser(description="Run Inference on Time-LLM Models")
    parser.add_argument("--type", choices=["teacher", "distilled"], help="Model type")
    parser.add_argument("--model", help="Model name (bert, distilbert, tinybert, etc.)")
    parser.add_argument("--dataset", default="584", help="Dataset identifier")
    parser.add_argument("--all", action="store_true", help="Run inference on all available models")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't run inference")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-results", action="store_true", help="List previous inference results")
    
    args = parser.parse_args()
    
    runner = InferenceRunner()
    
    if args.list_models:
        runner.list_available_models()
        return
        
    if args.list_results:
        runner.list_inference_results()
        return
    
    if args.all:
        runner.run_all_inference(args.dataset, args.dry_run)
    elif args.type and args.model:
        runner.run_inference(args.type, args.model, args.dataset, args.dry_run)
    else:
        print("Error: Specify --type and --model, or use --all for all models")
        parser.print_help()


if __name__ == "__main__":
    main()