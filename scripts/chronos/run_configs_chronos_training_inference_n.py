import import_utils
import os
import subprocess

from sympy import im

from utilities.extract_metrics import extract_metrics_to_csv
from extract_metrics_corrected import extract_corrected_metrics_to_csv
from fix_chronos import process_csv_files

import sys
import os



# Base directory where config files are stored
base_output_dir = "./experiment_configs_chronos_training_inference_missing_periodic/"
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_inference_missing_periodic.csv",
    )
process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_inference_missing_periodic_fixed.csv",
)

base_output_dir = "./experiment_configs_chronos_training_inference_missing_random/"
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_inference_missing_random.csv",
    )

process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_inference_missing_random_fixed.csv",
)

base_output_dir = "./experiment_configs_chronos_training_inference_noisy/"
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_inference_noisy.csv",
    )

process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_inference_noisy_fixed.csv",
)

# Base directory where config files are stored
base_output_dir = (
    "./experiment_configs_chronos_training_missing_periodic_inference_missing_periodic/"
)
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_missing_periodic_inference_missing_periodic.csv",
    )
process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_missing_periodic_inference_missing_periodic_fixed.csv",
)

base_output_dir = (
    "./experiment_configs_chronos_training_missing_random_inference_missing_random/"
)
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_missing_random_inference_missing_random.csv",
    )

process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_missing_random_inference_missing_random_fixed.csv",
)

base_output_dir = "./experiment_configs_chronos_training_noisy_inference_noisy/"
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
    extract_metrics_to_csv(
        base_dir=base_output_dir,
        output_csv="./experiment_configs_chronos_training_noisy_inference_noisy.csv",
    )


process_csv_files(base_output_dir)
extract_corrected_metrics_to_csv(
    base_output_dir,
    "./experiment_configs_chronos_training_noisy_inference_noisy_fixed.csv",
)


