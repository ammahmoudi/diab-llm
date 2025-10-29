# BG Data Prediction Using LLMs

This project aims to make predictions using Large Language Models (LLMs) with a dataset for time-series and inference tasks. Follow the instructions below to set up the environment and run the scripts to generate and execute configurations.

## Clone Repository

To clone this repository with all its submodules (including the Chronos forecasting model), use:

```bash
git clone --recursive https://github.com/PeterDomanski/LLM-TIME.git
```

Or if you've already cloned the repository, initialize and update the submodules:

```bash
git submodule update --init --recursive
```

## Quick Start

```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Install system dependencies for plotting (required for image generation)
sudo apt update && sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 \
  libpango-1.0-0 libcairo2 libasound2

# 3. Install Chronos for advanced time series forecasting (optional but recommended)
cd models/chronos && pip install --editable ".[training]" && cd ../..

# 4. Process your data (if needed)
python scripts/data_formatting/quick_process.py all

# 5A. Run basic experiments
python ./scripts/run_configs_time_llm_inference.py

# 5B. OR run knowledge distillation pipeline (recommended)
bash scripts/distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 1 --student-epochs 1 --distill-epochs 1 --dry-run
```

## 📊 Time-LLM Model Support Summary

The Time-LLM implementation supports **10 different language models** across the entire ecosystem:

### 🔬 **Core Models** (Production Ready)
- **🧠 BERT** `(bert-base-uncased)` - 768 dimensions, proven performance
- **⚡ DistilBERT** `(distilbert-base-uncased)` - 768 dimensions, 40% faster than BERT  
- **🏃 TinyBERT** `(huawei-noah/TinyBERT_General_4L_312D)` - 312 dimensions, ultra-fast inference
- **🎯 BERT-tiny** `(prajjwal1/bert-tiny)` - 128 dimensions, minimal resource usage

### 🚀 **Advanced Models** (Extended Support)
- **💡 MiniLM** `(microsoft/MiniLM-L12-H384-A12)` - 384 dimensions, efficient transformer
- **📱 MobileBERT** `(google/mobilebert-uncased)` - 512 dimensions, mobile-optimized
- **🎓 ALBERT** `(albert-base-v2)` - 768 dimensions, parameter sharing architecture  
- **🔤 GPT2** `(gpt2)` - 768 dimensions, generative capabilities
- **🌟 OPT-125M** `(facebook/opt-125m)` - 768 dimensions, Meta's optimized model
- **🦙 LLAMA** `(Various sizes)` - Research/experimental support

### 🔧 **Ecosystem Consistency**
✅ **Main Training**: All 10 models supported in `main.py`  
✅ **Distillation Pipeline**: Full model support in `distillation/scripts/`  
✅ **Config Generation**: Unified generator supports all models  
✅ **Run Scripts**: Automated execution for all model types  
✅ **Documentation**: Comprehensive guides for each model

### 📈 **Performance Characteristics**
| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| BERT-tiny | XS | ⚡⚡⚡⚡ | ⭐⭐⭐ | Quick prototyping |
| TinyBERT | S | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast inference |
| MiniLM | S-M | ⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| DistilBERT | M | ⚡⚡ | ⭐⭐⭐⭐ | Production standard |
| BERT | L | ⚡ | ⭐⭐⭐⭐⭐ | Benchmark reference |

> **Note**: All models are consistently mapped across training, distillation, and inference pipelines. Use the unified config generator for seamless model switching.

## 🧠 Knowledge Distillation Pipeline

**NEW**: Complete 3-phase knowledge distillation pipeline with multi-patient support and automatic CSV logging!

**📖 Full Documentation: [docs/DISTILLATION_README.md](docs/DISTILLATION_README.md)**

### 🔬 **NEW: Distillation Testing Framework**

Automatically find the best teacher-student pairs for your use case:

```bash
# Quick test (3 pairs, ~30 minutes)
./scripts/test_distillation_pairs.sh quick

# Strategic test (8 pairs, ~2 hours) - RECOMMENDED
./scripts/test_distillation_pairs.sh balanced

# Test single best pair (~20 minutes)
./scripts/test_distillation_pairs.sh best

# Test ultra-tiny models (~45 minutes)
./scripts/test_distillation_pairs.sh tiny
```

**📊 Automatic Analysis**: Results include performance comparison, rankings, and specific recommendations for your use case.

**📖 Detailed Guide: [docs/DISTILLATION_MODEL_PAIRS.md](docs/DISTILLATION_MODEL_PAIRS.md)**

### Quick Start Examples

**Single Patient**:
```bash
bash scripts/distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1
```

**Multiple Patients**:
```bash
bash scripts/distill_pipeline.sh \
  --teacher bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570,584 \
  --dataset ohiot1dm \
  --seed 42 \
  --teacher-epochs 1 \
  --student-epochs 1 \
  --distill-epochs 1
```

### 🤖 Supported Models

The project supports **10 different language models** ranging from large high-performance teacher models to small efficient student models:

**Large Teacher Models**:
- 🎯 **BERT** (110M) - Best general performance  
- 🚀 **GPT2** (117M) - Decoder-only architecture
- 🔥 **LLAMA** (6.7B) - High-capacity model
- ⚡ **DistilBERT** (66M) - Balanced size/performance

**Small Efficient Models**:
- 💎 **TinyBERT** (14M) - Purpose-built for distillation
- ⚡ **BERT-tiny** (4.4M) - Ultra-fast inference
- 🏃 **MiniLM** (33M) - Performance/size optimized
- 📱 **MobileBERT** (25M) - Mobile-optimized
- 🎯 **ALBERT** (12-18M) - Parameter sharing efficient
- 🚀 **OPT-125M** (125M) - Meta's efficient decoder

**📋 Full Model Details**: [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md)

### Features
- ✅ **Multi-Patient Support**: Process multiple patients with comma-separated IDs
- ✅ **3-Phase Pipeline**: Teacher training → Student baseline → Knowledge distillation  
- ✅ **Automatic CSV Logging**: Results saved after each patient completion
- ✅ **HuggingFace Integration**: Support for any compatible teacher/student models
- ✅ **Organized Output**: Timestamped directories with per-patient results
- ✅ **Performance Improvements**: Typically 3-8% RMSE improvement over teacher models

### Results Location
```bash
# All results automatically saved to:
distillation_experiments/pipeline_results.csv

# Individual pipeline runs:
distillation_experiments/pipeline_runs/pipeline_TIMESTAMP/patient_ID/
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

### 4. Install Chronos Time Series Forecasting (Training Mode)

For advanced time series forecasting capabilities, install the Chronos model in training mode:

```bash
# Navigate to the chronos directory and install in editable mode with training dependencies
cd models/chronos && pip install --editable ".[training]"
```

**What this provides:**
- 🧠 **Chronos Models**: State-of-the-art time series forecasting with T5-based architectures
- ⚡ **Chronos-Bolt**: Ultra-fast variants (250x faster, 20x more memory efficient)
- 🎯 **Multiple Model Sizes**: From tiny (8M params) to large (710M params)
- 🔧 **Training Capabilities**: Fine-tuning, pretraining, and research features
- 📊 **Zero-shot Forecasting**: Pretrained models work out-of-the-box

**Available Models:**
| Model | Parameters | Speed | Use Case |
|-------|------------|--------|----------|
| chronos-t5-tiny | 8M | ⚡⚡⚡⚡ | Quick testing |
| chronos-t5-small | 46M | ⚡⚡⚡ | Production ready |
| chronos-t5-base | 200M | ⚡⚡ | High accuracy |
| chronos-bolt-* | Various | ⚡⚡⚡⚡ | Ultra-fast inference |

**Usage Example:**
```python
import torch
from chronos import BaseChronosPipeline

# Load model
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small")

# Generate forecasts
quantiles, mean = pipeline.predict_quantiles(
    context=torch.tensor(your_time_series_data),
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9]
)
```

### 5. System Dependencies (Required for Plotting)

For plotting functionality with Kaleido/Chrome (required for result visualization), install these system packages:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2
```

**Chrome Installation (automatically handled by plotly_get_chrome):**
```bash
# Activate your virtual environment first
source venv/bin/activate
# Chrome will be automatically installed when you run experiments
plotly_get_chrome  # Optional: install Chrome manually
```

**Note:** The system will automatically attempt to install Chrome dependencies when running experiments. If you encounter plotting errors, ensure the above system packages are installed.

### 6. Path Configuration (Automatic)

The project uses dynamic path resolution and works automatically from any installation location. No manual path configuration is needed. All scripts automatically detect the project root and configure paths accordingly.

### 7. Process Your Data (Optional)

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
# Or use the convenience wrapper
python scripts/run_chronos.py

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

### Time-LLM Experiments (Unified System)

The project includes a comprehensive unified Time-LLM system with cross-scenario inference:

```bash
# Generate comprehensive configs (train on standardized, test on noisy)
python scripts/time_llm/config_generator_time_llm_unified.py --mode train_inference \
    --data_scenario noisy --train_data_scenario standardized \
    --patients 570,584 --llm_models GPT2,LLAMA --epochs 10

# Run all Time-LLM experiments
python scripts/time_llm/run_all_time_llm_experiments.py --modes train_inference --datasets ohiot1dm

# Generate all possible combinations automatically
python scripts/time_llm/generate_all_time_llm_configs.py
```

See `docs/README_time_llm_commands.md` for complete command reference.

### Legacy Time-LLM Scripts (Deprecated)

```bash
python ./scripts/run_configs_time_llm_inference.py
python ./scripts/run_configs_time_llm_training.py
```

For legacy scripts, you can change the order of models or filter them by type by editing the order list in the run_configs_time_llm_ .py files. Also for multi-GPU runs you can edit the run_main.sh file.

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
- **Automatic CSV logging**: All experiment results automatically saved to CSV files with metrics and configuration details

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

## Results & Metrics Extraction

### Automatic CSV Logging

All experiment runners automatically extract metrics to CSV files after each successful experiment:

- **Individual Results**: Each experiment type gets its own CSV file (e.g., `chronos_training_ohiot1dm_results.csv`)
- **Comprehensive Results**: All experiments combined into comprehensive CSV files
- **Real-time Updates**: CSV files updated after each individual experiment completion

### Manual Metrics Extraction

Extract all results into CSV files:

```bash
# Extract all metrics (both Chronos and Time-LLM)
python extract_all_metrics.py

# Extract only Chronos metrics
python extract_all_metrics.py --chronos_only

# Extract only Time-LLM metrics  
python extract_all_metrics.py --time_llm_only

# Save to custom directory
python extract_all_metrics.py --output_dir ./results
```

### CSV File Contents

Each CSV includes:
- **Configuration Details**: Model type, parameters, patient ID, seed, etc.
- **Performance Metrics**: RMSE, MAE, MAPE values
- **Experiment Metadata**: Timestamps, experiment type, data scenarios
- **Cross-scenario Information**: Training vs testing data scenarios

### Example CSV Structure
```csv
seed,model,dtype,mode,inference,pred_length,patient_id,log_datetime,rmse,mae,mape
831363,amazon-chronos-t5-base,float32,trained_inference,context,6,540,2025-10-11_16-16-47,195.52,46.91,NaN
```

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
- **Automatic CSV Logging**: Experiment results automatically extracted to CSV files after each run

---

## Troubleshooting

### Common Issues

#### Plotting/Visualization Errors
If you encounter errors related to Kaleido or Chrome during plotting:

```bash
# Error: "Kaleido requires Google Chrome to be installed"
# Solution: Install system dependencies
sudo apt update && sudo apt-get install -y libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2

# Error: "BrowserDepsError: missing common dependencies"
# Solution: Install Chrome via plotly_get_chrome
source venv/bin/activate
plotly_get_chrome

# Error: Plotly version compatibility
# Solution: Update plotly to compatible version
pip install "plotly>=6.1.1"
```

#### CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# For CUDA out of memory errors, reduce batch size in config files
# or use CPU mode by setting device: 'cpu' in configs
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x run_cross_scenario_inference.sh
chmod +x run_time_llm_bert_cross_scenario.sh

# Fix Python path issues
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
