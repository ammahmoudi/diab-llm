
# BG Data Prediction Using LLMs

This project aims to make predictions using Large Language Models (LLMs) with a dataset for time-series and inference tasks. Follow the instructions below to set up the environment and run the scripts to generate and execute configurations.

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

---

## Folder Structure

Before running the configuration scripts, ensure the following folders are available in the root directory and contain the necessary configuration files:

- `experiment_configs_time_llm_inference/` – Contains configurations for the time-series LLM  inference model.
- `experiment_configs_time_llm_training/` – Contains configurations for the time-series LLM  training+testing model.
- `experiment_configs_chronos_inference/` – Contains configurations for the Chronos inference model.

---

## Running Configurations

To run the configurations for both tasks, use the following commands:

```bash
python ./scripts/run_configs_time_llm_inference.py
python ./scripts/run_configs_time_llm_training.py
python ./scripts/run_configs_chronos_inference.py
```

These scripts will process all the configurations in the respective folders and save the results in a CSV file located in the root directory. For Time-LLM you can change the order of models or filter them by type by editing the order list in the run_configs_time_llm_ .py files. also for multi gpu runs you can edit the run_main.sh file.

---

## Generating Custom Configurations

If you wish to create your own custom configuration combinations, you can use the following scripts:

- `config_generator_chronos_inference.py` – For generating custom configurations for Chronos.
- `config_generator_chronos_training.py` – For generating custom configurations for Chronos.
- `config_generator_time_llm.py` – For generating custom configurations for the Time LLM model.

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

## Data Preprocessing

### 1. Standardize Raw Data Folder

To convert and standardize the raw dataset folder, run:

```bash
python ./scripts/format_files.py
```

### 2. Convert Data to Sequence & Prediction Length

For converting the data into sequence format and prediction-length rows, use the following script:

```bash
python ./scripts/reformat_bg_Data_folder.py
```

---
