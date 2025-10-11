
# BG Data Prediction Using LLMs

This project aims to make predictions using Large Language Models (LLMs) with a dataset for time-series and inference tasks. Follow the instructions below to set up the environment and run the scripts to generate and execute configurations.

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Process your data (if needed)
python scripts/data_formatting/quick_process.py all

# 3. Run experiments
python ./scripts/run_configs_time_llm_inference.py
```

## Setup Instructions

### 1. Create a Virtual Environment

Create a virtual environment called `venv`:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

To activate the virtual environment, use the following command:

```bash
source venv/bin/activate
```

### 3. Install Required Packages

After activating the virtual environment, install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Path Configuration (Automatic)

The project uses dynamic path resolution and works automatically from any installation location. No manual path configuration is needed. All scripts automatically detect the project root and configure paths accordingly.

### 5. Process Your Data (Optional)

If you have raw data that needs processing:

```bash
# Quick start - process all data
python scripts/data_formatting/quick_process.py all
```

See the **Data Processing Pipeline** section below for details.

---

## Folder Structure

Before running the configuration scripts, ensure the following folders are available in the root directory and contain the necessary configuration files:

- `experiment_configs_time_llm_inference/` – Contains configurations for the time-series LLM  inference model.
- `experiment_configs_time_llm_training/` – Contains configurations for the time-series LLM  training+testing model.
- `experiment_configs_chronos_inference/` – Contains configurations for the Chronos inference model.

---

## Running Experiments

### Chronos Experiments (Recommended)

The project now includes comprehensive Chronos model support with GPU acceleration and cross-scenario inference:

```bash
# Generate training configs
python scripts/chronos/config_generator_chronos.py --mode train --patients 570,584 --data_scenario standardized

# Run training with GPU acceleration
python scripts/chronos/run_all_chronos_experiments.py --modes training --datasets ohiot1dm

# Generate cross-scenario inference configs (train on standardized, test on noisy)
python scripts/chronos/config_generator_chronos.py --mode trained_inference --dataset ohiot1dm --data_scenario noisy --patients 570,584 --train_scenario standardized --window_config 6_6

# Run inference experiments
python scripts/chronos/run_all_chronos_experiments.py --modes trained_inference --datasets ohiot1dm
```

See `docs/README_chronos_commands.md` for complete command reference.

### Time-LLM Experiments

```bash
python ./scripts/run_configs_time_llm_inference.py
python ./scripts/run_configs_time_llm_training.py
```

For Time-LLM you can change the order of models or filter them by type by editing the order list in the run_configs_time_llm_ .py files. Also for multi-GPU runs you can edit the run_main.sh file.

---

## Generating Custom Configurations

### Chronos Configuration Generator (Unified)

The project includes a unified Chronos configuration generator supporting all modes:

```bash
# Training configs
python scripts/chronos/config_generator_chronos.py --mode train \
    --patients 570,584 --data_scenario standardized --models amazon/chronos-t5-base

# Inference configs (6_6 or 6_9 window configurations)
python scripts/chronos/config_generator_chronos.py --mode inference \
    --patients 570,584 --window_config 6_6

# Cross-scenario inference (train on one scenario, test on another)
python scripts/chronos/config_generator_chronos.py --mode trained_inference \
    --dataset ohiot1dm --data_scenario noisy --patients 570,584 \
    --train_scenario standardized --window_config 6_6

# LoRA fine-tuning
python scripts/chronos/config_generator_chronos.py --mode lora_inference \
    --patients 570,584 --use_lora
```

### Time-LLM Configuration Generator

```bash
python config_generator_time_llm.py  # For generating custom configurations for the Time LLM model
```

### Key Features:
- **Cross-scenario inference**: Train on one data scenario, test on another
- **Multiple window configurations**: 6_6 (6 input, 6 prediction) and 6_9 (6 input, 9 prediction)
- **GPU acceleration**: Automatic GPU detection and optimization
- **Comprehensive reporting**: Efficiency metrics, performance analysis, and result visualization

---

## Efficiency Benchmarking (For Reviewers)

### Quick Efficiency Report Generation

To generate comprehensive efficiency metrics that address reviewer requirements:

```bash
# Generate comprehensive efficiency report (RECOMMENDED)
python comprehensive_efficiency_report.py \
    --config experiment_configs_time_llm_inference/seed_809906_model_GPT2_dim_768_seq_6_context_6_pred_6_patch_6_epochs_0/patient_540/config.gin \
    --output-dir ./reviewer_results
```

This generates:
- **JSON report** with all metrics
- **LaTeX tables** ready for publication
- **Reviewer summary** addressing specific concerns

### Alternative Benchmarking Methods

```bash
# Simple benchmark (if model loading issues)
python simple_benchmark.py --model time_llm --config <config_path> --output-dir ./results

# Wrapper benchmark (uses existing main.py)
python wrapper_benchmark.py --config <config_path> --output-dir ./results --runs 3
```

### Install Efficiency Dependencies

```bash
pip install -r requirements_efficiency.txt
```

See `EFFICIENCY_BENCHMARKING_GUIDE.md` for detailed instructions.

---

## Data Processing Pipeline

### Complete Data Processing (Recommended)

Process all data in one command - standardization, formatting (6_6 and 6_9), and Arrow conversion:

```bash
# Process all datasets and scenarios
python scripts/data_formatting/quick_process.py all

# Process specific dataset  
python scripts/data_formatting/quick_process.py ohiot1dm
python scripts/data_formatting/quick_process.py d1namo

# Process specific scenarios only
python scripts/data_formatting/quick_process.py ohiot1dm --scenarios raw,noisy
```

### Advanced Pipeline Control

For more control over the processing pipeline:

```bash
# Full pipeline with all options
python scripts/data_formatting/complete_data_pipeline.py --dataset ohiot1dm --scenarios all

# Dry run to see what will be processed
python scripts/data_formatting/complete_data_pipeline.py --dataset ohiot1dm --dry-run

# Skip specific steps
python scripts/data_formatting/complete_data_pipeline.py --dataset ohiot1dm --skip-standardize
python scripts/data_formatting/complete_data_pipeline.py --dataset ohiot1dm --skip-format --skip-arrow
```

### Processing Steps

1. **Standardization** - Converts data to standard format (item_id, timestamp, target)
2. **Formatting** - Creates time series windows for both 6_6 and 6_9 configurations  
3. **Arrow Conversion** - Converts to Arrow format for efficient training

### Supported Data

- **Datasets**: ohiot1dm, d1namo
- **Scenarios**: raw, missing_periodic, missing_random, noisy, denoised
- **Window Configs**: 6_6 (input=6, pred=6), 6_9 (input=6, pred=9)
- **Output Formats**: CSV (formatted), Arrow (training-ready)

For detailed documentation, see `scripts/data_formatting/README.md`.

---

## Key Features & Improvements

### 🚀 GPU Acceleration
- **Automatic GPU Detection**: All models automatically detect and use available GPUs
- **Memory Optimization**: RTX 2060 optimized configurations with proper memory management
- **Training Acceleration**: Up to 100% GPU utilization during Chronos training
- **Inference Optimization**: GPU-accelerated inference with batch processing

### 🔄 Cross-Scenario Inference
- **Robustness Testing**: Train on one data scenario, test on another (e.g., train on clean data, test on noisy data)
- **Scenario Support**: standardized, noisy, denoised, missing_periodic, missing_random
- **Automatic Checkpoint Discovery**: System automatically finds and loads appropriate trained models
- **Performance Analysis**: Comprehensive metrics comparing model performance across scenarios

### 📊 Comprehensive Performance Monitoring
- **Real-time Profiling**: GPU memory, utilization, temperature, and power consumption tracking
- **Efficiency Metrics**: Model parameters (201M+ for Chronos-base), memory usage, inference speed
- **Edge Feasibility Analysis**: Automatic assessment for edge deployment suitability
- **Detailed Reports**: JSON and visual reports with publication-ready metrics

### 🗃️ Advanced Data Processing
- **Unified Data Pipeline**: Single command to process all datasets and scenarios
- **Multiple Window Configurations**: Support for 6_6 and 6_9 time series windows
- **Arrow Format Support**: Efficient training data format with 10x faster loading
- **Dynamic Path Resolution**: Works from any installation location without configuration

### 🧠 Model Improvements
- **Chronos Integration**: Full support for Amazon's Chronos foundation models
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning capabilities
- **Multiple Model Sizes**: Support for tiny, base, and large Chronos variants
- **Checkpoint Management**: Automatic checkpoint saving, loading, and cleanup

---
