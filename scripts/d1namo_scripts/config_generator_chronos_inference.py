import os
import json
from itertools import product
import random
from seeds import fixed_seeds

# Define parameter sets that must be consistent
feature_label_sets = [
    {
        "input_features": [
            "BG_{t-5}",
            "BG_{t-4}",
            "BG_{t-3}",
            "BG_{t-2}",
            "BG_{t-1}",
            "BG_{t}",
        ],
        "labels": [
            "BG_{t+1}",
            "BG_{t+2}",
            "BG_{t+3}",
            "BG_{t+4}",
            "BG_{t+5}",
            "BG_{t+6}",
        ],
        "prediction_length": 6,
        "context_length": 6,
    },
    {
        "input_features": [
            "BG_{t-5}",
            "BG_{t-4}",
            "BG_{t-3}",
            "BG_{t-2}",
            "BG_{t-1}",
            "BG_{t}",
        ],
        "labels": [
            "BG_{t+1}",
            "BG_{t+2}",
            "BG_{t+3}",
            "BG_{t+4}",
            "BG_{t+5}",
            "BG_{t+6}",
            "BG_{t+7}",
            "BG_{t+8}",
            "BG_{t+9}",
        ],
        "prediction_length": 9,
        "context_length": 6,
    },
]

# Define the parameters to iterate over
patients = ["001",
            "002", "003","004","005", "006", "007"
            ] # List of patient IDs
# seeds = [2021, 2022]  # List of seeds


# Generate 10 random seeds between 0 and 999999 (or any range you like)
# seeds = [random.randint(0, 999999) for _ in range(10)]
seeds = fixed_seeds
# print(seeds)


models = [
    "amazon/chronos-t5-tiny",
    # "amazon/chronos-t5-mini",
    # "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
    # "amazon/chronos-t5-large",
]  # Different models
torch_dtypes = [
    # "float16",
    # "bfloat16",
    "float32",
]  # Different torch dtypes
modes = ["inference"]  # Different modes


# Base directory for configurations and logs
base_output_dir = "./experiment_d1namo_configs_chronos_inference/"
os.makedirs(base_output_dir, exist_ok=True)

# Generate config files for all combinations
for seed, feature_label_set, model, torch_dtype, mode in product(
    seeds, feature_label_sets, models, torch_dtypes, modes
):
    context_len = feature_label_set["context_length"]
    pred_len = feature_label_set["prediction_length"]

    # Set min_past to match context_length
    min_past = context_len

    # Define a unique folder for this configuration
    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{model.replace('/', '-').replace('_','-')}_dtype_{torch_dtype}_mode_{mode}_context_{context_len}_pred_{pred_len}",
    )
    os.makedirs(config_folder, exist_ok=True)

    for patient_id in patients:
        # Create a patient-specific subfolder within the config group
        patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        # Define log directory inside the same patient folder
        log_folder = os.path.join(patient_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        # Define the dynamic data path using context and prediction lengths
        data_folder = f"./data/d1namo_formatted/{context_len}_{pred_len}"

        # Prepare .gin configuration content
        config_content = f"""
run.log_dir = "{log_folder}"
run.chronos_dir = "/home/amma/"

run.data_settings = {{
    'path_to_train_data': '{data_folder}/{patient_id}-ws-training.csv',
    'path_to_test_data': '{data_folder}/{patient_id}-ws-testing.csv',
    'input_features': {feature_label_set["input_features"]},
    'labels': {feature_label_set["labels"]},
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'percent': 100
}}

run.llm_settings = {{
    'mode': '{mode}',    
    'method': 'chronos',    
    'model': '{model}',  
    'torch_dtype': '{torch_dtype}',   
    'prediction_length': {pred_len},    
    'num_samples': 1,
    'context_length': {context_len},
    'min_past': {min_past},
    'prediction_batch_size': 64,    
    'prediction_use_auto_split': False,
    'eval_metrics': ['rmse', 'mae', 'mape'],    
    'restore_from_checkpoint': False,
    'restore_checkpoint_path': '',
    'seed': {seed}  
}}
"""

        # Save .gin config file inside the patient-specific folder
        config_filename = "config.gin"
        config_path = os.path.join(patient_folder, config_filename)

        with open(config_path, "w") as f:
            f.write(config_content.strip())

        print(f"Generated: {config_path} with logs stored in {log_folder}")
