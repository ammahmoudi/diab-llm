#!/usr/bin/env python3
"""
Flexible Experiment Runner for Time-LLM
Runs experiments with full control over datasets, data types, patients, and models.
"""

import os
import sys
import subprocess
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add utils to path for path utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.path_utils import get_project_root


class FlexibleExperimentRunner:
    """Run experiments with flexible configuration control."""
    
    def __init__(self, base_dir=None, output_dir=None):
        if base_dir is None:
            base_dir = get_project_root()
        self.base_dir = Path(base_dir)
        if output_dir:
            self.configs_dir = Path(output_dir) / "configs"
            self.results_dir = Path(output_dir) / "results"
        else:
            self.configs_dir = self.base_dir / "distillation_experiments" / "configs" 
            self.results_dir = self.base_dir / "distillation_experiments" / "summaries"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        
        # Import the config generator
        sys.path.append(str(self.base_dir / "scripts"))
        from flexible_config_generator import FlexibleConfigGenerator
        self.config_generator = FlexibleConfigGenerator(configs_dir=str(self.configs_dir))

    def run_single_experiment(self, config_path, dry_run=False):
        """Run a single experiment from config file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        experiment_name = config_path.stem.replace('config_', '')
        
        print(f"\n{'='*60}")
        print(f"Running Experiment: {experiment_name}")
        print(f"Config: {config_path}")
        print(f"{'='*60}")
        
        if dry_run:
            print("DRY RUN: Would execute experiment")
            return {"experiment_name": experiment_name, "status": "dry_run", "config_path": str(config_path)}
        
        # Execute training command
        cmd = [
            "python", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Change to base directory for execution
        original_dir = os.getcwd()
        start_time = datetime.now()
        
        try:
            os.chdir(self.base_dir)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                print(f"✓ Experiment completed successfully in {duration:.2f} seconds")
                
                # Extract metrics from output if available
                metrics = self.extract_metrics_from_output(result.stdout)
                
                # Create experiment summary
                summary = self.create_experiment_summary(
                    experiment_name, config_path, "success", duration, metrics, result.stdout
                )
                
                return summary
            else:
                print(f"✗ Experiment failed with return code: {result.returncode}")
                print("STDERR:", result.stderr)
                
                summary = self.create_experiment_summary(
                    experiment_name, config_path, "failed", duration, {}, 
                    result.stdout, result.stderr, result.returncode
                )
                
                return summary
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"✗ Experiment failed with exception: {str(e)}")
            
            summary = self.create_experiment_summary(
                experiment_name, config_path, "error", duration, {}, "", str(e)
            )
            
            return summary
        finally:
            os.chdir(original_dir)

    def extract_metrics_from_output(self, output):
        """Extract evaluation metrics from command output."""
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            # Look for various metric formats
            if any(metric in line.lower() for metric in ["rmse", "mae", "mape", "mse"]):
                try:
                    if ":" in line:
                        parts = line.strip().split(':')
                        if len(parts) == 2:
                            metric_name = parts[0].strip().lower()
                            metric_value = float(parts[1].strip())
                            metrics[metric_name] = metric_value
                except:
                    continue
        
        return metrics

    def create_experiment_summary(self, experiment_name, config_path, status, duration, 
                                metrics=None, stdout="", stderr="", return_code=None):
        """Create a summary of the experiment run."""
        summary = {
            "experiment_name": experiment_name,
            "config_path": str(config_path),
            "status": status,
            "duration_seconds": duration,
            "start_time": datetime.now().isoformat(),
            "metrics": metrics or {},
            "return_code": return_code
        }
        
        # Save detailed logs if available
        if stdout or stderr:
            log_dir = self.results_dir / "logs" / experiment_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if stdout:
                with open(log_dir / "stdout.log", 'w') as f:
                    f.write(stdout)
                summary["stdout_log"] = str(log_dir / "stdout.log")
            
            if stderr:
                with open(log_dir / "stderr.log", 'w') as f:
                    f.write(stderr)
                summary["stderr_log"] = str(log_dir / "stderr.log")
        
        # Save experiment summary
        summary_file = self.results_dir / f"{experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Experiment summary saved: {summary_file}")
        return summary

    def run_batch_experiments(self, batch_dir_or_pattern, dry_run=False, parallel=False):
        """Run multiple experiments from a batch directory or pattern."""
        
        if Path(batch_dir_or_pattern).is_dir():
            # Run all configs in directory
            config_files = list(Path(batch_dir_or_pattern).glob("config_*.gin"))
        else:
            # Use as glob pattern - handle both relative and absolute paths
            pattern_path = Path(batch_dir_or_pattern)
            if pattern_path.is_absolute():
                # For absolute paths, use parent directory and glob the filename pattern
                parent_dir = pattern_path.parent
                pattern_name = pattern_path.name
                config_files = list(parent_dir.glob(pattern_name))
            else:
                config_files = list(Path(batch_dir_or_pattern).glob("*.gin"))
        
        if not config_files:
            print(f"No configuration files found in: {batch_dir_or_pattern}")
            return []
        
        print(f"\n{'='*80}")
        print(f"RUNNING BATCH EXPERIMENTS")
        print(f"Found {len(config_files)} configurations")
        print(f"Dry Run: {dry_run}")
        print(f"{'='*80}")
        
        results = []
        failed_experiments = []
        
        for i, config_file in enumerate(config_files, 1):
            print(f"\n[{i}/{len(config_files)}] Processing: {config_file.name}")
            
            try:
                result = self.run_single_experiment(config_file, dry_run)
                results.append(result)
                
                if result["status"] == "failed" or result["status"] == "error":
                    failed_experiments.append(result["experiment_name"])
                    
            except Exception as e:
                print(f"✗ Failed to run experiment {config_file.name}: {str(e)}")
                failed_experiments.append(config_file.stem)
        
        # Create batch summary
        batch_summary = self.create_batch_summary(results, batch_dir_or_pattern)
        
        print(f"\n{'='*80}")
        print(f"BATCH EXPERIMENTS COMPLETE")
        print(f"Total: {len(results)}, Successful: {len(results) - len(failed_experiments)}, Failed: {len(failed_experiments)}")
        if failed_experiments:
            print(f"Failed experiments: {', '.join(failed_experiments)}")
        print(f"{'='*80}")
        
        return batch_summary

    def create_batch_summary(self, results, batch_identifier):
        """Create a summary of batch experiment results."""
        batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        summary = {
            "batch_name": batch_name,
            "batch_identifier": str(batch_identifier),
            "run_time": datetime.now().isoformat(),
            "total_experiments": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "errors": len([r for r in results if r["status"] == "error"]),
            "experiments": results
        }
        
        # Save batch summary
        batch_file = self.results_dir / f"{batch_name}_summary.json"
        with open(batch_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create comparison report if metrics are available
        self.create_batch_comparison_report(results, batch_name)
        
        print(f"✓ Batch summary saved: {batch_file}")
        return summary

    def create_batch_comparison_report(self, results, batch_name):
        """Create a comparison report for batch results."""
        comparison_data = []
        
        for result in results:
            if result["status"] == "success" and result.get("metrics"):
                row = {
                    "experiment_name": result["experiment_name"],
                    "status": result["status"],
                    "duration": result["duration_seconds"]
                }
                row.update(result["metrics"])
                comparison_data.append(row)
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            report_file = self.results_dir / f"{batch_name}_comparison.csv"
            df.to_csv(report_file, index=False)
            print(f"✓ Comparison report saved: {report_file}")
            
            # Print top performers
            if "rmse" in df.columns:
                print("\nTop 5 performers (by RMSE):")
                top_5 = df.nsmallest(5, "rmse")[["experiment_name", "rmse"]]
                print(top_5.to_string(index=False))

    def generate_and_run(self, dataset, data_type, patient_ids, model_names, 
                        epochs=20, learning_rate=0.001, dry_run=False):
        """Generate configurations and run experiments in one step."""
        
        print(f"\n{'='*80}")
        print(f"GENERATE AND RUN EXPERIMENTS")
        print(f"Dataset: {dataset}, Data Type: {data_type}")
        print(f"Patients: {patient_ids}, Models: {model_names}")
        print(f"{'='*80}")
        
        # Generate all combinations
        experiments_spec = []
        for patient_id in patient_ids:
            for model_name in model_names:
                experiments_spec.append({
                    "dataset": dataset,
                    "data_type": data_type,
                    "patient_id": patient_id,
                    "model_name": model_name,
                    "train_epochs": epochs,
                    "learning_rate": learning_rate
                })
        
        # Create batch configurations
        batch_name = f"generated_{dataset}_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_summary = self.config_generator.create_experiment_batch(experiments_spec, batch_name)
        
        # Run the batch
        batch_dir = self.configs_dir / batch_name
        return self.run_batch_experiments(batch_dir, dry_run)

    def list_available_configs(self):
        """List available configuration files."""
        print("\nAvailable Configuration Files:")
        print("-" * 50)
        
        # Single configs
        single_configs = list(self.configs_dir.glob("config_*.gin"))
        if single_configs:
            print("\nSingle Configurations:")
            for config in sorted(single_configs):
                print(f"  - {config.name}")
        
        # Batch configs
        batch_dirs = [d for d in self.configs_dir.iterdir() if d.is_dir()]
        if batch_dirs:
            print("\nBatch Configurations:")
            for batch_dir in sorted(batch_dirs):
                config_count = len(list(batch_dir.glob("config_*.gin")))
                print(f"  - {batch_dir.name}/ ({config_count} configs)")

    def list_experiment_results(self):
        """List previous experiment results."""
        print("\nExperiment Results:")
        print("-" * 50)
        
        # Individual experiments
        individual_results = list(self.results_dir.glob("*_summary.json"))
        batch_results = list(self.results_dir.glob("batch_*_summary.json"))
        
        if individual_results:
            print(f"\nIndividual Experiments ({len(individual_results)}):")
            for result in sorted(individual_results)[-10:]:  # Show last 10
                with open(result, 'r') as f:
                    data = json.load(f)
                status = data.get('status', 'unknown')
                duration = data.get('duration_seconds', 0)
                print(f"  - {result.stem}: {status} ({duration:.2f}s)")
        
        if batch_results:
            print(f"\nBatch Experiments ({len(batch_results)}):")
            for result in sorted(batch_results)[-5:]:  # Show last 5
                with open(result, 'r') as f:
                    data = json.load(f)
                total = data.get('total_experiments', 0)
                successful = data.get('successful', 0)
                print(f"  - {result.stem}: {successful}/{total} successful")


def main():
    parser = argparse.ArgumentParser(description="Flexible Experiment Runner for Time-LLM")
    
    # Single experiment
    parser.add_argument("--config", help="Path to single configuration file")
    
    # Batch experiments
    parser.add_argument("--batch", help="Path to batch directory or glob pattern")
    
    # Generate and run
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--data-type", help="Data type")
    parser.add_argument("--patients", nargs="+", help="Patient IDs")
    parser.add_argument("--models", nargs="+", help="Model names")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    
    # Options
    parser.add_argument("--dry-run", action="store_true", help="Generate configs but don't run")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel (future)")
    parser.add_argument("--output-dir", help="Output directory for experiments")
    
    # Information
    parser.add_argument("--list-configs", action="store_true", help="List available configurations")
    parser.add_argument("--list-results", action="store_true", help="List experiment results")
    
    args = parser.parse_args()
    
    runner = FlexibleExperimentRunner(output_dir=args.output_dir)
    
    if args.list_configs:
        runner.list_available_configs()
        return
    
    if args.list_results:
        runner.list_experiment_results()
        return
    
    if args.config:
        runner.run_single_experiment(args.config, args.dry_run)
    
    elif args.batch:
        runner.run_batch_experiments(args.batch, args.dry_run, args.parallel)
    
    elif args.dataset and args.data_type and args.patients and args.models:
        runner.generate_and_run(
            args.dataset, args.data_type, args.patients, args.models,
            args.epochs, args.lr, args.dry_run
        )
    
    else:
        print("Error: Specify --config, --batch, or generate-and-run parameters")
        parser.print_help()


if __name__ == "__main__":
    main()