#!/usr/bin/env python3
"""
Comprehensive Efficiency Testing Runner
========================================

This script runs efficiency tests for all major model types:
1. Time-LLM models: BERT, GPT2, LLAMA (training + inference)
2. Chronos models: T5-base, T5-tiny (training + inference) 
3. Distillation pipeline: BERT→TinyBERT (training + inference)

For efficiency analysis, we test with:
- One patient (570 for ohiot1dm)
- One seed (831363)
- Standardized data scenario
- Full train+inference cycle to capture both training and inference efficiency

The script automatically:
- Generates configurations using unified config generators
- Runs experiments with efficiency monitoring enabled
- Collects comprehensive efficiency reports (memory, GPU, timing, power)
- Organizes results in structured folders for analysis

Usage:
    python comprehensive_efficiency_runner.py [--dry-run]
    
Examples:
    python comprehensive_efficiency_runner.py --dry-run          # Preview focused experiments (16)
    python comprehensive_efficiency_runner.py                    # Run focused experiments
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
# Fixed seeds for consistency
FIXED_SEEDS = [831363, 809906, 427368, 238822, 247659]

class ComprehensiveEfficiencyRunner:
    """Comprehensive efficiency testing for all model types."""
    
    def __init__(self, base_dir=None, dry_run=False):
        """Initialize the efficiency runner."""
        if base_dir is None:
            base_dir = os.getcwd()
        self.base_dir = Path(base_dir)
        self.dry_run = dry_run
        
        # Test parameters for efficiency focus
        self.test_patient = "570"  # Single patient for efficiency testing
        self.test_seed = str(FIXED_SEEDS[0])  # Use first seed: 831363
        self.data_scenario = "standardized"  # Clean data for consistent efficiency measurement
        self.dataset = "ohiot1dm"  # Primary dataset
        
        # Model configurations - Using separate modes for cleaner efficiency measurement
        self.models_config = {
            "time_llm": {
                "script": "scripts/time_llm/config_generator_time_llm_unified.py",
                "models": ["BERT", "GPT2", "LLAMA"],
                "modes": {
                    "train": {"epochs": 10, "mode": "train"},
                    "inference": {"epochs": 0, "mode": "inference"}
                },
                "window_config": "6_9"  # 6 context, 9 prediction
            },
            "chronos": {
                "script": "scripts/chronos/config_generator_chronos.py", 
                "models": ["amazon/chronos-t5-base", "amazon/chronos-t5-tiny"],
                "modes": {
                    "train": {"mode": "train"},
                    "inference": {"mode": "inference"}
                },
                "window_config": "6_9"  # 6 context, 9 prediction
            },
            "distillation": {
                "script": "distill_pipeline.sh",
                "pairs": [("bert-base-uncased", "prajjwal1/bert-tiny")],  # BERT → TinyBERT
                "teacher_epochs": 10,
                "student_epochs": 10, 
                "distill_epochs": 10
            }
        }
        
        # Output directory for efficiency experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / f"efficiency_experiments_{timestamp}"
        
        print(f"🔧 Comprehensive Efficiency Testing Setup")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📊 Test patient: {self.test_patient}")
        print(f"🎲 Test seed: {self.test_seed}")
        print(f"📈 Data scenario: {self.data_scenario}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🎯 Comprehensive Efficiency Testing Setup")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"🔍 Dry run: {self.dry_run}")
        
        # Fixed seeds and focused testing parameters
        self.test_patient = "570"
        self.test_seed = str(FIXED_SEEDS[0])  # Use first seed: 831363
        self.data_scenario = "standardized" 
        self.dataset = "ohiot1dm"
        
    def run_command(self, cmd, description, cwd=None, timeout_hours=6):
        """Run a command with proper error handling and logging for long experiments."""
        if cwd is None:
            cwd = self.base_dir
            
        print(f"\n{'='*60}")
        print(f"🚀 {description}")
        print(f"📂 Working directory: {cwd}")
        print(f"⚡ Command: {cmd}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("🔍 DRY RUN - Command would be executed here")
            return True
        
        # For long experiments, we need to show live output and have longer timeout
        timeout_seconds = timeout_hours * 3600
        
        try:
            # Run command with live output (no capture_output for long experiments)
            if "run_all" in cmd or "distill_pipeline" in cmd:
                print(f"🏃 Starting long-running experiment (timeout: {timeout_hours}h)...")
                print(f"📺 Live output will be shown below:")
                print("-" * 60)
                
                # Run without capturing output so we can see what's happening
                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=cwd,
                    timeout=timeout_seconds,
                    text=True
                )
                
                print("-" * 60)
                if result.returncode == 0:
                    print(f"✅ SUCCESS: {description}")
                    return True
                else:
                    print(f"❌ FAILED: {description}")
                    print(f"💥 Error code: {result.returncode}")
                    return False
            else:
                # For quick commands (config generation), capture output
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    cwd=cwd,
                    capture_output=True, 
                    text=True,
                    timeout=300  # 5 minutes for config generation
                )
                
                if result.returncode == 0:
                    print(f"✅ SUCCESS: {description}")
                    if result.stdout.strip():
                        print(f"📝 Output:\n{result.stdout}")
                    return True
                else:
                    print(f"❌ FAILED: {description}")
                    print(f"💥 Error code: {result.returncode}")
                    if result.stderr.strip():
                        print(f"🚨 Error output:\n{result.stderr}")
                    if result.stdout.strip():
                        print(f"📝 Standard output:\n{result.stdout}")
                    return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: {description} exceeded {timeout_hours} hour limit")
            return False
        except Exception as e:
            print(f"💥 EXCEPTION: {description} failed with: {str(e)}")
            return False
    

    
    def run_generated_experiments(self, experiment_type):
        """Run the generated experiment configurations using proper experiment runners."""
        print(f"\n🏃 RUNNING {experiment_type.upper()} EFFICIENCY EXPERIMENTS")
        print(f"{'='*80}")
        
        results = []
        
        if experiment_type == "time_llm":
            # Use the Time-LLM experiment runner with specific filters for our efficiency configs
            cmd = [
                "python", "scripts/time_llm/run_all_time_llm_experiments.py",
                "--modes", "train,inference", 
                "--datasets", self.dataset,
                "--models", "BERT,GPT2,LLAMA",
                "--log_level", "INFO",
                # Force a fresh run by removing any resume file
                "--resume_file", f"time_llm_efficiency_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ]
            
            cmd_str = " ".join(cmd)
            print(f"🚀 Time-LLM Command: {cmd_str}")
            
            # Important: Remove any existing resume files to force fresh execution
            resume_files = list(self.base_dir.glob("time_llm_experiments_progress.json"))
            for resume_file in resume_files:
                print(f"🗑️ Removing old resume file: {resume_file}")
                resume_file.unlink(missing_ok=True)
            
            # Wrap with efficiency monitoring
            efficiency_cmd = [
                "python", "efficiency/real_time_profiler.py",
                "--output_file", f"time_llm_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "--"
            ] + cmd
            
            efficiency_cmd_str = " ".join(efficiency_cmd)
            description = f"Run Time-LLM experiments with efficiency monitoring (BERT, GPT2, LLAMA)"
            success = self.run_command(efficiency_cmd_str, description)
            results.append(("time_llm_all_models", success))
            
        elif experiment_type == "chronos":
            # Use the Chronos experiment runner with specific filters for our efficiency configs
            cmd = [
                "python", "scripts/chronos/run_all_chronos_experiments.py", 
                "--modes", "training,inference",
                "--datasets", self.dataset,
                "--log_level", "INFO",
                # Force a fresh run by removing any resume file
                "--resume_file", f"chronos_efficiency_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            ]
            
            cmd_str = " ".join(cmd)
            print(f"⏰ Chronos Command: {cmd_str}")
            
            # Important: Remove any existing resume files to force fresh execution
            resume_files = list(self.base_dir.glob("chronos_experiments_progress.json"))
            for resume_file in resume_files:
                print(f"🗑️ Removing old resume file: {resume_file}")
                resume_file.unlink(missing_ok=True)
            
            # Wrap with efficiency monitoring
            efficiency_cmd = [
                "python", "efficiency/real_time_profiler.py",
                "--output_file", f"chronos_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "--"
            ] + cmd
            
            efficiency_cmd_str = " ".join(efficiency_cmd)
            description = f"Run Chronos experiments with efficiency monitoring (T5-base, T5-tiny)"
            success = self.run_command(efficiency_cmd_str, description)
            results.append(("chronos_all_models", success))
        
        return results
    
    def collect_efficiency_reports(self):
        """Collect and summarize efficiency reports from all experiments."""
        print(f"\n📊 COLLECTING EFFICIENCY REPORTS")
        print(f"{'='*80}")
        
        # Find all efficiency reports
        report_patterns = [
            "**/efficiency_report_*.json",
            "**/real_performance_report_*.json", 
            "**/comprehensive_performance_report_*.json"
        ]
        
        all_reports = []
        for pattern in report_patterns:
            reports = list(self.base_dir.glob(pattern))
            all_reports.extend(reports)
        
        print(f"📋 Found {len(all_reports)} efficiency reports")
        
        # Group reports by experiment type
        report_groups = {
            "time_llm": [],
            "chronos": [],
            "distillation": []
        }
        
        for report in all_reports:
            report_path = str(report)
            if "time_llm" in report_path:
                report_groups["time_llm"].append(report)
            elif "chronos" in report_path:
                report_groups["chronos"].append(report)
            elif "distillation" in report_path:
                report_groups["distillation"].append(report)
        
        # Create summary
        summary_file = self.output_dir / "efficiency_summary.txt"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE EFFICIENCY TESTING SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test Configuration:\n")
            f.write(f"- Patient: {self.test_patient}\n")
            f.write(f"- Seed: {self.test_seed}\n")
            f.write(f"- Data Scenario: {self.data_scenario}\n")
            f.write(f"- Dataset: {self.dataset}\n\n")
            
            for exp_type, reports in report_groups.items():
                f.write(f"{exp_type.upper()} EFFICIENCY REPORTS:\n")
                f.write("-" * 40 + "\n")
                for report in reports:
                    rel_path = report.relative_to(self.base_dir)
                    f.write(f"  {rel_path}\n")
                f.write(f"Total: {len(reports)} reports\n\n")
        
        print(f"📝 Efficiency summary saved to: {summary_file}")
        return all_reports
    

    
    def find_focused_configs(self):
        """Find specific configs for focused efficiency testing."""
        configs = []
        
        # Time-LLM configs (training and inference)
        time_llm_patterns = [
            f"experiments/time_llm_training_{self.dataset}/seed_{self.test_seed}_model_*_epochs_10/patient_{self.test_patient}/config.gin",
            f"experiments/time_llm_inference_{self.dataset}/seed_{self.test_seed}_model_*_epochs_0/patient_{self.test_patient}/config.gin"
        ]
        
        for pattern in time_llm_patterns:
            matches = list(self.base_dir.glob(pattern))
            for config_file in matches:
                if config_file.exists():
                    configs.append({
                        'type': 'time_llm',
                        'path': config_file,
                        'relative_path': config_file.relative_to(self.base_dir)
                    })
        
        # Chronos configs (training and inference)
        chronos_patterns = [
            f"experiments/chronos_training_{self.dataset}/seed_{self.test_seed}_model_*_mode_train_*/patient_{self.test_patient}/config.gin",
            f"experiments/chronos_inference_{self.dataset}/seed_{self.test_seed}_model_*_mode_inference_*/patient_{self.test_patient}/config.gin"
        ]
        
        for pattern in chronos_patterns:
            matches = list(self.base_dir.glob(pattern))
            for config_file in matches:
                if config_file.exists():
                    configs.append({
                        'type': 'chronos',
                        'path': config_file,
                        'relative_path': config_file.relative_to(self.base_dir)
                    })
        
        return configs
    

    
    def run_single_experiment(self, config_path, experiment_type):
        """Run a single experiment with efficiency monitoring."""
        print(f"\n{'='*60}")
        print(f"🚀 Running {experiment_type.upper()} Efficiency Experiment")
        print(f"📄 Config: {config_path.relative_to(self.base_dir)}")
        print(f"{'='*60}")
        
        if self.dry_run:
            print("🔍 DRY RUN - Would execute experiment here")
            return True
        
        # Command to run the experiment directly (no efficiency wrapper)
        cmd = [
            "python3", "main.py",
            "--config_path", str(config_path),
            "--log_level", "INFO"
        ]
        
        cmd_str = " ".join(cmd)
        print(f"⚡ Command: {cmd_str}")
        print(f"🏃 Starting experiment...")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, timeout=7200)  # 2 hour timeout
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: Completed in {duration:.2f}s")
                return True
            else:
                print(f"❌ FAILED: Exit code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ TIMEOUT: Experiment exceeded 2 hour limit")
            return False
        except Exception as e:
            print(f"💥 ERROR: {str(e)}")
            return False
    
    def run_focused_experiments(self):
        """Run focused efficiency tests on specific configs."""
        print(f"\n🎯 STARTING FOCUSED EFFICIENCY TESTING")
        print(f"{'='*80}")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Test patient: {self.test_patient}")
        print(f"🎲 Test seed: {self.test_seed}")
        print(f"📈 Data scenario: {self.data_scenario}")
        
        # Find our specific configs
        configs = self.find_focused_configs()
        
        if not configs:
            print("❌ No focused test configs found!")
            print("💡 The focused configs should be generated automatically.")
            print("💡 Make sure you have configs in the experiments/ directory:")
            print("   - experiments/time_llm_training_ohiot1dm/")
            print("   - experiments/time_llm_inference_ohiot1dm/") 
            print("   - experiments/chronos_training_ohiot1dm/")
            print("   - experiments/chronos_inference_ohiot1dm/")
            print("💡 Run the config generators first if needed.")
            return False
        
        print(f"\n📋 Found {len(configs)} focused test configs:")
        for config in configs:
            print(f"  {config['type']}: {config['relative_path']}")
        
        if self.dry_run:
            print("\n🔍 DRY RUN MODE - No experiments will be executed")
            return True
        
        # Run each experiment
        results = {'success': [], 'failed': []}
        for config in configs:
            success = self.run_single_experiment(config['path'], config['type'])
            if success:
                results['success'].append(config['relative_path'])
            else:
                results['failed'].append(config['relative_path'])
        
        # Summary
        total = len(configs)
        success_count = len(results['success'])
        
        print(f"\n🎯 FOCUSED EFFICIENCY TESTING COMPLETE")
        print(f"{'='*80}")
        print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ Successful: {success_count}/{total}")
        print(f"❌ Failed: {len(results['failed'])}/{total}")
        
        if results['failed']:
            print(f"\n❌ Failed experiments:")
            for failure in results['failed']:
                print(f"  {failure}")
        
        return success_count == total

def main():
    """Main entry point for the comprehensive efficiency runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Efficiency Testing for LLM-TIME Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_efficiency_runner.py --dry-run          # Preview focused experiments (16)
  python comprehensive_efficiency_runner.py                    # Run focused experiments
        """
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview experiments without executing them"
    )
    

    
    args = parser.parse_args()
    
    # Always run focused efficiency testing
    runner = ComprehensiveEfficiencyRunner(dry_run=args.dry_run)
    success = runner.run_focused_experiments()
    return 0 if success else 1
    
    # Check if any experiments failed
    failed_experiments = []
    for phase, phase_results in results.items():
        if isinstance(phase_results, list):
            for exp_name, success in phase_results:
                if not success:
                    failed_experiments.append(exp_name)
    
    if failed_experiments:
        print(f"\n⚠️  Some experiments failed: {failed_experiments}")
        return 1
    else:
        print(f"\n🎉 All efficiency tests completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())