import os
import subprocess
from extract_metrics import extract_metrics_to_csv

# Base directory where config files are stored
base_output_dir = "./d1namo_experiment_configs_time_llm_inference/"
log_level = "INFO"

# Model priority order
model_order = ["GPT2","BERT"]

# Recursively find all `config.gin` files and categorize by model type
config_files_by_model = {"GPT2": [], "BERT": [], "LLAMA": []}

for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_path = os.path.join(root, file)
            if "GPT2" in config_path.upper():
                config_files_by_model["GPT2"].append(config_path)
            elif "BERT" in config_path.upper():
                config_files_by_model["BERT"].append(config_path)
            elif "LLAMA" in config_path.upper():
                config_files_by_model["LLAMA"].append(config_path)

# Run `run_main.sh` for each config in order of model priority
for model in model_order:
    for config_path in config_files_by_model[model]:
        command = f"./run_main.sh --config_path {config_path} --log_level {log_level} --remove_checkpoints True"
        print(f"Running: {command}")
        subprocess.run(command, shell=True)
        extract_metrics_to_csv(base_dir=base_output_dir, output_csv='./d1namo_experiment_results_time_llm_inference.csv')

# After running the main scripts, extract metrics
