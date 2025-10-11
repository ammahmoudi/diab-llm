#!/usr/bin/env python3
"""
Unified Chronos Experiment Runner

This script runs ALL Chronos experiments from generated configuration files.
It automatically discovers and executes experiments across:
- All datasets (d1namo, ohiot1dm)
- All scenarios (standardized, missing_periodic, missing_random, noisy, denoised)
- All modes (training, inference, cross-scenario)
- All patients and model combinations

Features:
- Automatic config discovery from experiments folder
- Parallel execution support (optional)
- Full console output visibility for training logs
- Progress tracking and logging
- Automatic metrics extraction
- Resume capability (skip completed experiments)
- Configurable execution parameters

Usage Examples:
    # Run all experiments with full console output
    python run_all_chronos_experiments.py
    
    # Run only training experiments (recommended)
    python run_all_chronos_experiments.py --modes training
    
    # Run only OhioT1DM experiments
    python run_all_chronos_experiments.py --modes training --datasets ohiot1dm
    
    # Run with parallel execution (hides individual logs)
    python run_all_chronos_experiments.py --modes training --parallel --max_workers 2
    
    # Resume from previous run
    python run_all_chronos_experiments.py --modes training --resume
    
    # Dry run (show what would be executed)
    python run_all_chronos_experiments.py --modes training --dry_run
"""

import os
import sys
import glob
import argparse
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def find_config_files(base_dir, mode_filter=None, dataset_filter=None):
    """Find all config.gin files in the experiments directory."""
    config_files = []
    
    # Pattern to match Chronos experiment directories
    chronos_pattern = os.path.join(base_dir, "chronos_*")
    
    for exp_dir in glob.glob(chronos_pattern):
        exp_name = os.path.basename(exp_dir)
        
        # Apply filters
        if mode_filter:
            if not any(mode in exp_name for mode in mode_filter):
                continue
                
        if dataset_filter:
            if not any(dataset in exp_name for dataset in dataset_filter):
                continue
        
        # Find all config.gin files in this experiment directory
        for root, _, files in os.walk(exp_dir):
            for file in files:
                if file == "config.gin":
                    config_path = os.path.join(root, file)
                    config_files.append({
                        'path': config_path,
                        'experiment': exp_name,
                        'relative_path': os.path.relpath(config_path, exp_dir)
                    })
    
    return config_files

def is_experiment_completed(config_info, resume_file=None):
    """Check if an experiment has already been completed."""
    if resume_file and os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            completed = json.load(f)
            return config_info['path'] in completed.get('completed', [])
    return False

def mark_experiment_completed(config_path, resume_file):
    """Mark an experiment as completed."""
    completed_data = {'completed': [], 'failed': []}
    
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            completed_data = json.load(f)
    
    if config_path not in completed_data['completed']:
        completed_data['completed'].append(config_path)
    
    with open(resume_file, 'w') as f:
        json.dump(completed_data, f, indent=2)

def mark_experiment_failed(config_path, resume_file, error_msg=""):
    """Mark an experiment as failed."""
    completed_data = {'completed': [], 'failed': []}
    
    if os.path.exists(resume_file):
        with open(resume_file, 'r') as f:
            completed_data = json.load(f)
    
    if config_path not in completed_data['failed']:
        completed_data['failed'].append({
            'path': config_path,
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        })
    
    with open(resume_file, 'w') as f:
        json.dump(completed_data, f, indent=2)

def run_single_experiment(config_info, log_level="INFO", remove_checkpoints=None, dry_run=False, resume_file=None):
    """Run a single experiment configuration."""
    config_path = config_info['path']
    experiment_name = config_info['experiment']
    
    if resume_file and is_experiment_completed(config_info, resume_file):
        print(f"⏭️  Skipping completed experiment: {config_path}")
        return True, f"Skipped (already completed): {config_path}"
    
    # Determine remove_checkpoints based on experiment type
    if remove_checkpoints is None:
        # For training, keep checkpoints; for inference, remove them
        remove_checkpoints = "inference" in experiment_name.lower()
    
    command = [
        "python", "main.py",
        "--config_path", config_path,
        "--log_level", log_level,
        "--remove_checkpoints", str(remove_checkpoints).lower()
    ]
    
    print(f"🚀 Running: {' '.join(command)}")
    print(f"   📂 Experiment: {experiment_name}")
    print(f"   📄 Config: {config_info['relative_path']}")
    
    if dry_run:
        print(f"   🔍 DRY RUN - Command would be executed")
        return True, f"Dry run: {config_path}"
    
    try:
        start_time = time.time()
        result = subprocess.run(command, capture_output=False, text=True, timeout=3600)  # 1 hour timeout, show output
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Completed in {duration:.2f}s: {config_path}")
            if resume_file:
                mark_experiment_completed(config_path, resume_file)
            
            # Extract metrics to CSV after successful completion
            try:
                from scripts.utilities.extract_metrics import extract_metrics_to_csv
                experiment_base_dir = os.path.dirname(os.path.dirname(config_path))  # Go up to experiment folder
                csv_filename = f"chronos_{experiment_name}_results.csv"
                output_csv = os.path.join("./", csv_filename)  # Save in root directory
                extract_metrics_to_csv(base_dir=experiment_base_dir, output_csv=output_csv)
                print(f"   📊 Metrics extracted to: {output_csv}")
            except Exception as e:
                print(f"   ⚠️  Metrics extraction failed: {e}")
            
            return True, f"Success ({duration:.2f}s): {config_path}"
        else:
            error_msg = f"Exit code {result.returncode}"
            print(f"   ❌ Failed: {config_path}")
            print(f"   📋 Command: {' '.join(command)}")
            print(f"   🔍 Exit Code: {result.returncode}")
            if result.stdout:
                print(f"   📤 STDOUT:\n{result.stdout[:500]}")
            if result.stderr:
                print(f"   📥 STDERR:\n{result.stderr[:500]}")
            if resume_file:
                mark_experiment_failed(config_path, resume_file, error_msg)
            return False, f"Failed: {config_path} - {error_msg}"
            
    except subprocess.TimeoutExpired:
        error_msg = "Timeout after 1 hour"
        print(f"   ⏰ Timeout: {config_path}")
        if resume_file:
            mark_experiment_failed(config_path, resume_file, error_msg)
        return False, f"Timeout: {config_path}"
    except Exception as e:
        error_msg = str(e)
        print(f"   💥 Exception: {config_path} - {error_msg}")
        if resume_file:
            mark_experiment_failed(config_path, resume_file, error_msg)
        return False, f"Exception: {config_path} - {error_msg}"

def extract_metrics_for_experiment(experiment_dir):
    """Extract metrics for a completed experiment."""
    try:
        from utilities.extract_metrics import extract_metrics_to_csv
        csv_file = os.path.join(experiment_dir, "experiment_results.csv")
        extract_metrics_to_csv(base_dir=experiment_dir, output_csv=csv_file)
        print(f"   📊 Metrics extracted: {csv_file}")
        return True
    except Exception as e:
        print(f"   ⚠️  Metrics extraction failed: {e}")
        return False

def run_experiments_parallel(config_files, max_workers=2, **kwargs):
    """Run experiments in parallel."""
    results = {'success': [], 'failed': []}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_config = {
            executor.submit(run_single_experiment, config, **kwargs): config 
            for config in config_files
        }
        
        # Collect results
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                success, message = future.result()
                if success:
                    results['success'].append(message)
                else:
                    results['failed'].append(message)
            except Exception as e:
                error_msg = f"Parallel execution error: {config['path']} - {e}"
                results['failed'].append(error_msg)
                print(f"💥 {error_msg}")
    
    return results

def run_experiments_sequential(config_files, **kwargs):
    """Run experiments sequentially."""
    results = {'success': [], 'failed': []}
    
    for config in config_files:
        success, message = run_single_experiment(config, **kwargs)
        if success:
            results['success'].append(message)
        else:
            results['failed'].append(message)
    
    return results

def main():
    # Change to project root directory for consistent paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)
    
    parser = argparse.ArgumentParser(description="Unified Chronos Experiment Runner")
    parser.add_argument("--experiments_dir", default="./experiments", 
                       help="Base directory containing experiment configurations")
    parser.add_argument("--modes", default=None,
                       help="Comma-separated list of experiment modes to run (training,inference,cross_scenario)")
    parser.add_argument("--datasets", default=None,
                       help="Comma-separated list of datasets to run (d1namo,ohiot1dm)")
    parser.add_argument("--log_level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--remove_checkpoints", default=None, type=bool,
                       help="Whether to remove checkpoints (None=auto)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run experiments in parallel")
    parser.add_argument("--max_workers", type=int, default=2,
                       help="Maximum parallel workers")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run")
    parser.add_argument("--resume_file", default="chronos_experiments_progress.json",
                       help="File to track completed experiments")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be run without executing")
    parser.add_argument("--extract_metrics", action="store_true", default=True,
                       help="Extract metrics after running experiments (default: True)")
    parser.add_argument("--no_extract_metrics", action="store_true",
                       help="Disable metrics extraction")
    
    args = parser.parse_args()
    
    # Handle metrics extraction flags
    if args.no_extract_metrics:
        args.extract_metrics = False
    
    # Parse filter arguments
    mode_filter = args.modes.split(',') if args.modes else None
    dataset_filter = args.datasets.split(',') if args.datasets else None
    
    print("🔍 Discovering Chronos experiment configurations...")
    print(f"📁 Base directory: {os.path.abspath(args.experiments_dir)}")
    
    # Find all config files
    config_files = find_config_files(
        args.experiments_dir, 
        mode_filter=mode_filter,
        dataset_filter=dataset_filter
    )
    
    if not config_files:
        print("❌ No Chronos configuration files found!")
        print("   Make sure you've run the config generators first:")
        print("   python scripts/data_formatting/generate_all_chronos_configs.py")
        return 1
    
    print(f"🎯 Found {len(config_files)} experiment configurations")
    
    # Group by experiment type for summary
    experiments_by_type = {}
    for config in config_files:
        exp_type = config['experiment']
        if exp_type not in experiments_by_type:
            experiments_by_type[exp_type] = 0
        experiments_by_type[exp_type] += 1
    
    print("\n📊 Experiments by type:")
    for exp_type, count in sorted(experiments_by_type.items()):
        print(f"   {exp_type}: {count} configs")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No experiments will be executed")
    
    if args.resume and os.path.exists(args.resume_file):
        with open(args.resume_file, 'r') as f:
            progress = json.load(f)
            completed_count = len(progress.get('completed', []))
            failed_count = len(progress.get('failed', []))
            print(f"\n🔄 Resume mode: {completed_count} completed, {failed_count} failed previously")
    
    print(f"\n🚀 Starting experiment execution...")
    start_time = time.time()
    
    # Run experiments
    run_kwargs = {
        'log_level': args.log_level,
        'remove_checkpoints': args.remove_checkpoints,
        'dry_run': args.dry_run,
        'resume_file': args.resume_file if args.resume else None
    }
    
    if args.parallel and not args.dry_run:
        print(f"⚡ Running in parallel with {args.max_workers} workers")
        results = run_experiments_parallel(config_files, args.max_workers, **run_kwargs)
    else:
        print("🔄 Running sequentially")
        results = run_experiments_sequential(config_files, **run_kwargs)
    
    # Summary
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("CHRONOS EXPERIMENT EXECUTION COMPLETE!")
    print("="*60)
    print(f"⏱️  Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"📈 Success rate: {len(results['success'])/(len(results['success'])+len(results['failed']))*100:.1f}%")
    
    if results['failed']:
        print("\n❌ Failed experiments:")
        for failure in results['failed'][:10]:  # Show first 10 failures
            print(f"   {failure}")
        if len(results['failed']) > 10:
            print(f"   ... and {len(results['failed']) - 10} more")
    
    # Extract comprehensive metrics if requested
    if args.extract_metrics and not args.dry_run:
        print("\n📊 Extracting comprehensive metrics from all completed experiments...")
        try:
            from scripts.utilities.extract_metrics import extract_metrics_to_csv
            comprehensive_csv = "./chronos_comprehensive_results.csv"
            extract_metrics_to_csv(base_dir=args.experiments_dir, output_csv=comprehensive_csv)
            print(f"📊 Comprehensive metrics saved to: {comprehensive_csv}")
        except Exception as e:
            print(f"⚠️  Comprehensive metrics extraction failed: {e}")
        
        # Also extract individual experiment metrics
        for exp_type in experiments_by_type.keys():
            exp_dir = os.path.join(args.experiments_dir, exp_type)
            if os.path.exists(exp_dir):
                extract_metrics_for_experiment(exp_dir)
    
    print(f"\n🎉 Chronos experiment execution finished!")
    return 0 if not results['failed'] else 1

if __name__ == "__main__":
    exit(main())