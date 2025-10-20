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
    python comprehensive_efficiency_runner.py [--dry-run] [--models model1,model2,...]
    
Examples:
    python comprehensive_efficiency_runner.py --dry-run                    # Preview all commands
    python comprehensive_efficiency_runner.py --models time_llm            # Run only Time-LLM models
    python comprehensive_efficiency_runner.py --models chronos             # Run only Chronos models
    python comprehensive_efficiency_runner.py --models distillation        # Run only distillation
    python comprehensive_efficiency_runner.py                              # Run everything
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from utilities.seeds import fixed_seeds

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
        self.test_seed = str(fixed_seeds[0])  # Use first seed: 831363
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
                }
            },
            "chronos": {
                "script": "scripts/chronos/config_generator_chronos.py", 
                "models": ["amazon/chronos-t5-base", "amazon/chronos-t5-tiny"],
                "modes": {
                    "train": {"mode": "train"},
                    "inference": {"mode": "inference"}
                }
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
        print(f"🔍 Dry run: {self.dry_run}")
        
    def run_command(self, cmd, description, cwd=None):
        """Run a command with proper error handling and logging."""
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
            
        try:
            # Run command and capture output
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=cwd,
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout per command
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
            print(f"⏰ TIMEOUT: {description} exceeded 1 hour limit")
            return False
        except Exception as e:
            print(f"💥 EXCEPTION: {description} failed with: {str(e)}")
            return False
    
    def generate_time_llm_configs(self):
        """Generate Time-LLM configurations for efficiency testing (separate train and inference)."""
        print(f"\n🤖 GENERATING TIME-LLM EFFICIENCY CONFIGURATIONS")
        print(f"{'='*80}")
        
        results = []
        config = self.models_config["time_llm"]
        
        for llm_model in config["models"]:
            for mode_name, mode_config in config["modes"].items():
                cmd = (
                    f"python {config['script']} "
                    f"--mode {mode_config['mode']} "
                    f"--dataset {self.dataset} "
                    f"--data_scenario {self.data_scenario} "
                    f"--patients {self.test_patient} "
                    f"--llm_models {llm_model} "
                    f"--seeds {self.test_seed} "
                    f"--epochs {mode_config['epochs']}"
                )
                
                description = f"Generate Time-LLM {llm_model} {mode_name} efficiency config"
                success = self.run_command(cmd, description)
                results.append((f"time_llm_{llm_model.lower()}_{mode_name}", success))
            
        return results
    
    def generate_chronos_configs(self):
        """Generate Chronos configurations for efficiency testing (separate train and inference)."""
        print(f"\n⏰ GENERATING CHRONOS EFFICIENCY CONFIGURATIONS")
        print(f"{'='*80}")
        
        results = []
        config = self.models_config["chronos"]
        
        for chronos_model in config["models"]:
            model_name = chronos_model.replace("/", "_").replace("-", "_")
            
            for mode_name, mode_config in config["modes"].items():
                cmd = (
                    f"python {config['script']} "
                    f"--mode {mode_config['mode']} "
                    f"--dataset {self.dataset} "
                    f"--data_scenario {self.data_scenario} "
                    f"--patients {self.test_patient} "
                    f"--models {chronos_model} "
                    f"--seeds {self.test_seed}"
                )
                
                description = f"Generate Chronos {chronos_model} {mode_name} efficiency config" 
                success = self.run_command(cmd, description)
                results.append((f"chronos_{model_name}_{mode_name}", success))
            
        return results
    
    def run_distillation_efficiency(self):
        """Run distillation pipeline for efficiency testing."""
        print(f"\n🧠 RUNNING DISTILLATION EFFICIENCY PIPELINE")
        print(f"{'='*80}")
        
        results = []
        config = self.models_config["distillation"]
        
        for teacher, student in config["pairs"]:
            cmd = (
                f"bash {config['script']} "
                f"--teacher {teacher} "
                f"--student {student} "
                f"--patients {self.test_patient} "
                f"--dataset {self.dataset} "
                f"--seed {self.test_seed} "
                f"--teacher-epochs {config['teacher_epochs']} "
                f"--student-epochs {config['student_epochs']} "
                f"--distill-epochs {config['distill_epochs']}"
            )
            
            pair_name = f"{teacher}_to_{student}".replace("/", "_").replace("-", "_")
            description = f"Run distillation efficiency: {teacher} → {student}"
            success = self.run_command(cmd, description)
            results.append((f"distillation_{pair_name}", success))
            
        return results
    
    def run_generated_experiments(self, experiment_type):
        """Run the generated experiment configurations."""
        print(f"\n🏃 RUNNING {experiment_type.upper()} EFFICIENCY EXPERIMENTS")
        print(f"{'='*80}")
        
        results = []
        
        # Find experiment directories 
        if experiment_type == "time_llm":
            patterns = [
                f"experiments/time_llm_train_{self.dataset}_{self.data_scenario}",
                f"experiments/time_llm_inference_{self.dataset}_{self.data_scenario}"
            ]
            main_script = "main.py"
        elif experiment_type == "chronos":
            patterns = [
                f"experiments/chronos_train_{self.dataset}_{self.data_scenario}",
                f"experiments/chronos_inference_{self.dataset}_{self.data_scenario}"
            ]
            main_script = "main.py"
        else:
            # Distillation is handled separately
            return results
            
        # Check all patterns
        experiment_dirs = []
        for pattern in patterns:
            experiment_dirs.extend(list(self.base_dir.glob(pattern)))
        
        for exp_dir in experiment_dirs:
            if not exp_dir.is_dir():
                continue
                
            # Find config files in subdirectories
            config_files = list(exp_dir.rglob("config.gin"))
            
            for config_file in config_files:
                if f"patient_{self.test_patient}" in str(config_file):
                    # Run the experiment
                    cmd = f"python {main_script} --config {config_file}"
                    
                    rel_path = config_file.relative_to(self.base_dir)
                    description = f"Run {experiment_type} efficiency experiment: {rel_path}"
                    success = self.run_command(cmd, description)
                    
                    exp_name = f"{experiment_type}_{config_file.parent.parent.name}"
                    results.append((exp_name, success))
        
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
    
    def run_efficiency_tests(self, model_types=None):
        """Run comprehensive efficiency tests for specified model types."""
        if model_types is None:
            model_types = ["time_llm", "chronos", "distillation"]
        
        print(f"\n🚀 STARTING COMPREHENSIVE EFFICIENCY TESTING")
        print(f"{'='*80}")
        print(f"📋 Model types to test: {', '.join(model_types)}")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_results = {}
        
        # Phase 1: Generate configurations
        if "time_llm" in model_types:
            all_results["time_llm_config"] = self.generate_time_llm_configs()
        
        if "chronos" in model_types:
            all_results["chronos_config"] = self.generate_chronos_configs()
        
        # Phase 2: Run experiments (if not dry run)
        if not self.dry_run:
            if "time_llm" in model_types:
                all_results["time_llm_experiments"] = self.run_generated_experiments("time_llm")
            
            if "chronos" in model_types:
                all_results["chronos_experiments"] = self.run_generated_experiments("chronos")
        
        # Phase 3: Run distillation (separate pipeline)
        if "distillation" in model_types:
            all_results["distillation"] = self.run_distillation_efficiency()
        
        # Phase 4: Collect reports
        if not self.dry_run:
            efficiency_reports = self.collect_efficiency_reports()
            all_results["reports_collected"] = len(efficiency_reports)
        
        # Summary
        self.print_final_summary(all_results)
        
        return all_results
    
    def print_final_summary(self, results):
        """Print a comprehensive summary of all efficiency tests."""
        print(f"\n🎯 COMPREHENSIVE EFFICIENCY TESTING SUMMARY")
        print(f"{'='*80}")
        print(f"⏰ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        total_success = 0
        total_attempted = 0
        
        for phase, phase_results in results.items():
            if isinstance(phase_results, list):
                print(f"\n📊 {phase.upper().replace('_', ' ')}:")
                for exp_name, success in phase_results:
                    status = "✅" if success else "❌"
                    print(f"  {status} {exp_name}")
                    total_attempted += 1
                    if success:
                        total_success += 1
            elif isinstance(phase_results, int):
                print(f"\n📋 {phase.upper().replace('_', ' ')}: {phase_results}")
        
        if total_attempted > 0:
            success_rate = (total_success / total_attempted) * 100
            print(f"\n🎉 OVERALL SUCCESS RATE: {total_success}/{total_attempted} ({success_rate:.1f}%)")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"  1. 📊 Use the experiment_efficiency_analysis.ipynb notebook to analyze results")
        print(f"  2. 📁 Check the experiments/ folder for efficiency reports")
        print(f"  3. 🔍 Look for *_performance_report_*.json files with detailed metrics")
        print(f"  4. 📈 Compare memory usage, latency, and power consumption across models")
        
        if self.dry_run:
            print(f"\n🔍 This was a DRY RUN - no actual experiments were executed")
            print(f"💡 Remove --dry-run flag to run the actual efficiency tests")

def main():
    """Main entry point for the comprehensive efficiency runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Efficiency Testing for LLM-TIME Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python comprehensive_efficiency_runner.py --dry-run
  python comprehensive_efficiency_runner.py --models time_llm
  python comprehensive_efficiency_runner.py --models chronos,distillation
  python comprehensive_efficiency_runner.py
        """
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Preview commands without executing them"
    )
    
    parser.add_argument(
        "--models",
        default="time_llm,chronos,distillation",
        help="Comma-separated list of model types to test (time_llm, chronos, distillation)"
    )
    
    args = parser.parse_args()
    
    # Parse model types
    if args.models:
        model_types = [m.strip() for m in args.models.split(",")]
        valid_types = ["time_llm", "chronos", "distillation"]
        model_types = [m for m in model_types if m in valid_types]
        if not model_types:
            print("❌ No valid model types specified. Valid options: time_llm, chronos, distillation")
            return 1
    else:
        model_types = ["time_llm", "chronos", "distillation"]
    
    # Create and run efficiency tester
    runner = ComprehensiveEfficiencyRunner(dry_run=args.dry_run)
    results = runner.run_efficiency_tests(model_types=model_types)
    
    # Return success code based on results
    if args.dry_run:
        return 0
    
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