import os
import subprocess
from extract_metrics import extract_metrics_to_csv

# Base directory where config files are stored
base_output_dir = "./experiment_configs_time_llm/"
log_level = "INFO"

# Recursively find all `config.gin` files
config_files = []
for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_files.append(os.path.join(root, file))

# Run `run_main.sh` for each config instead of calling main.py directly
for config_path in config_files:
    # Prepare the command to run the .sh script with the necessary arguments
    command = f"./run_main.sh --config_path {config_path} --log_level {log_level}"
    
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

# After running the main scripts, extract metrics
extract_metrics_to_csv(base_dir=base_output_dir, output_csv='./experiment_results_time_llm.csv')
