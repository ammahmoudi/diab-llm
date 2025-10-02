import import_utils
import os
import sys
import subprocess

from utilities.extract_metrics import extract_metrics_to_csv

# Base directory where config files are stored
base_output_dir = "./experiment_configs_time_llm_training/"
log_level = "INFO"

# Model priority order
model_order = [
    # "BERT",
    # "DistilBERT",
    # "GPT2",
    # "LLAMA",
    # "MiniLM",
    # "TinyBERT",
    # "MobileBERT",
    # "ALBERT",
    # "BERT-tiny",
    # "OPT-125M",
    "Chronos",
]

# Initialize dict dynamically
config_files_by_model = {model: [] for model in model_order}

# Recursively find all `config.gin` files and categorize by model type
for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_path = os.path.join(root, file)
            # Check model type dynamically
            for model in model_order:
                if model.upper() in config_path.upper():
                    config_files_by_model[model].append(config_path)

# Run `run_main.sh` for each config in order of model priority
for model in model_order:
    for config_path in config_files_by_model[model]:
        command = f"./run_main.sh --config_path {config_path} --log_level {log_level} --remove_checkpoints False"
        print(f"Running: {command}")
        subprocess.run(command, shell=True)
        extract_metrics_to_csv(base_dir=base_output_dir, output_csv='./experiment_results_time_llm_training.csv')

# After running the main scripts, extract metrics
