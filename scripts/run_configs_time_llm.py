import os
import subprocess

# Base directory where config files are stored
base_output_dir = "./experiment_configs_time_llm/"
log_level = "DEBUG"

# Recursively find all `config.gin` files
config_files = []
for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_files.append(os.path.join(root, file))

# Run `main.py` for each config
for config_path in config_files:
    command = f"python ./main.py --config_path {config_path} --log_level {log_level}"
    
    print(f"Running: {command}")
    subprocess.run(command, shell=True)
