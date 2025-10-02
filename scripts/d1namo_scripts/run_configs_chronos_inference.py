import import_utils
import os
import subprocess

from utilities.extract_metrics import extract_metrics_to_csv

import sys
import os


# Base directory where config files are stored
base_output_dir = "./experiment_d1namo_configs_chronos_inference/"
log_level = "DEBUG"

# Recursively find all `config.gin` files
config_files = []
for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_files.append(os.path.join(root, file))

# Run `main.py` for each config
for config_path in config_files:
    command = f"python ./main.py --config_path {config_path} --log_level {log_level} --remove_checkpoints=True"
    
    print(f"Running: {command}")
    subprocess.run(command, shell=True)
    extract_metrics_to_csv(base_dir=base_output_dir,output_csv='./expriment_d1namo_results_chronos_inference.csv')
