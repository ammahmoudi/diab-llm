import os
import json
from itertools import product

# Define parameter sets that must be consistent
feature_label_sets = [
    {
        "input_features": ["BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"],
        "labels": ["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"],
        "prediction_length": 6,
        "context_length": 6
    },
    {
        "input_features": [ "BG_{t-8}", "BG_{t-7}", "BG_{t-6}", "BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"],
        "labels": ["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}", "BG_{t+7}", "BG_{t+8}", "BG_{t+9}"],
        "prediction_length": 9,
        "context_length": 9
    }
]

# Define the parameters to iterate over
patients = ["540", "544", "552","559","563","567","570","575","584","588","591","596"]  # List of patient IDs
seeds = [2021, 2022]  # List of seeds
models = ["amazon/chronos-t5-tiny","amazon/chronos-t5-mini","amazon/chronos-t5-small", "amazon/chronos-t5-base","amazon/chronos-t5-large"]  # Different models
torch_dtypes = ["float16","bfloat16", "float32"]  # Different torch dtypes
modes = ["inference"]  # Different modes

# Define max_train_steps manually for each model (or any criteria you want)
max_train_steps_mapping = {
    "amazon/chronos-t5-tiny": 10000,
     "amazon/chronos-t5-mini": 10000,
      "amazon/chronos-t5-small": 10000,
    "amazon/chronos-t5-base": 10000,
     "amazon/chronos-t5-large": 10000,
}

# Base directory for configurations and logs
base_output_dir = "./experiment_configs/"
os.makedirs(base_output_dir, exist_ok=True)

# Generate config files for all combinations
for seed, feature_label_set, model, torch_dtype, mode in product(seeds, feature_label_sets, models, torch_dtypes, modes):
    context_len = feature_label_set["context_length"]
    pred_len = feature_label_set["prediction_length"]
    
    # Set min_past to match context_length
    min_past = context_len

    # Get max_train_steps from mapping
    max_train_steps = max_train_steps_mapping.get(model, 10000)  # Default to 10k if model is missing

    # Define a unique folder for this configuration
    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{model.replace('/', '_')}_dtype_{torch_dtype}_mode_{mode}_context_{context_len}_pred_{pred_len}"
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
        data_folder = f"./data/formatted/{context_len}_{pred_len}"

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
    'ntokens': 4096,
    'tokenizer_kwargs': "{{'low_limit': 35,'high_limit': 500}}",
    'prediction_length': {pred_len},    
    'num_samples': 1,
    'context_length': {context_len},
    'min_past': {min_past},
    'prediction_batch_size': 64,    
    'prediction_use_auto_split': False,
    'max_train_steps': {max_train_steps},
    'train_batch_size': 32,
    'random_init': False,
    'save_steps': 1000,
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