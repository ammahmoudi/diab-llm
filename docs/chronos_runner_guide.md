# Chronos Experiment Runner Guide

## Overview
The unified Chronos experiment runner discovers and executes **4,370 total configurations** across:

- **8 Cross-scenario experiments** (1,680 configs)
- **9 Inference experiments** (2,070 configs) 
- **6 Training experiments** (620 configs)

## Quick Usage

### Run All Experiments
```bash
# Run everything (4,370 experiments)
python run_chronos.py

# Run with parallel execution (recommended)
python run_chronos.py --parallel --max_workers 4
```

### Run Specific Types
```bash
# Run only training experiments
python run_chronos.py --modes training

# Run only inference experiments  
python run_chronos.py --modes inference

# Run specific datasets
python run_chronos.py --datasets d1namo
python run_chronos.py --datasets ohiot1dm
python run_chronos.py --datasets d1namo,ohiot1dm
```

### Resume and Progress Tracking
```bash
# Resume from previous run (skips completed experiments)
python run_chronos.py --resume

# Check what would run without executing
python run_chronos.py --dry_run

# Run with metrics extraction
python run_chronos.py --extract_metrics
```

### Advanced Options
```bash
# Parallel execution with custom workers
python run_chronos.py --parallel --max_workers 8

# Custom log level
python run_chronos.py --log_level DEBUG

# Run specific experiment combinations
python run_chronos.py --modes training,inference --datasets d1namo
```

## Experiment Breakdown

### Training (620 configs)
- chronos_training: 120 configs (ohiot1dm standardized)
- chronos_training_d1namo: 70 configs (d1namo standardized)
- chronos_training_*_denoised: 190 configs (both datasets)
- chronos_training_*_missing_periodic: 190 configs
- chronos_training_*_missing_random: 190 configs  
- chronos_training_*_noisy: 190 configs

### Inference (2,070 configs)
- chronos_inference: 240 configs (ohiot1dm standardized)
- chronos_inference_d1namo: 140 configs (d1namo standardized)
- chronos_inference_*_denoised: 380 configs (both datasets)
- chronos_inference_*_missing_periodic: 480 configs
- chronos_inference_*_missing_random: 480 configs
- chronos_inference_*_noisy: 480 configs

### Cross-Scenario (1,680 configs)  
- chronos_cross_scenario_d1namo_*: 560 configs (4 scenarios × 140 each)
- chronos_cross_scenario_ohiot1dm_*: 960 configs (4 scenarios × 240 each)

## Features
- **Automatic Discovery**: Finds all config.gin files in experiments/
- **Progress Tracking**: Saves progress in `chronos_experiments_progress.json`
- **Resume Capability**: Skip completed experiments on restart
- **Parallel Execution**: Run multiple experiments simultaneously
- **Filtering**: Run specific modes, datasets, or combinations
- **Metrics Extraction**: Auto-extract results after completion
- **Error Handling**: Continues on failures, tracks failed experiments
- **Dry Run**: Preview what will be executed without running

## Example Session
```bash
# Start with a small test
python run_chronos.py --modes training --datasets d1namo --dry_run

# Run the small test for real  
python run_chronos.py --modes training --datasets d1namo

# Run all training experiments in parallel
python run_chronos.py --modes training --parallel --max_workers 4

# Resume and run inference experiments  
python run_chronos.py --modes inference --resume --parallel
```

The script automatically handles checkpoint management, determines appropriate `--remove_checkpoints` settings, and provides detailed progress reporting.