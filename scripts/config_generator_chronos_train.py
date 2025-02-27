import os
import random
from itertools import product
from seeds import fixed_seeds

# Define parameter sets that must be consistent (only change for this new config is the patients' data path)
feature_label_sets = [
    {
        "input_features": ["target"],
        "labels": ["target"],
        "prediction_length": 64,
        "context_length": 512,
    },
]

# Define the patients (you can adjust this list as needed)
patients = [
    "540",
    "544",
    "552",
    "559",
    "563",
    "567",
    "570",
    "575",
    "584",
    "588",
    "591",
    "596",
]

# Generate 10 random seeds between 0 and 999999 (or any range you like)
# seeds = [random.randint(0, 999999) for _ in range(1)]
seeds=fixed_seeds

# Available models
models = [
    # "amazon/chronos-t5-tiny",
    # "amazon/chronos-t5-mini",
    # "amazon/chronos-t5-small",
    "amazon/chronos-t5-base",
    # "amazon/chronos-t5-large",
]

# Torch dtypes
torch_dtypes = [
    "bfloat16",
    # "float32",
]

# Modes (always 'training' in this case)
modes = ["training"]

# Max train steps for each model
max_train_steps_mapping = {
    # "amazon/chronos-t5-tiny": 10000,
    # "amazon/chronos-t5-mini": 10000,
    # "amazon/chronos-t5-small": 10000,
    "amazon/chronos-t5-base": 200000,
    # "amazon/chronos-t5-large": 200000,
}

# Base output directory
base_output_dir = "./experiment_configs_chronos_training/"
os.makedirs(base_output_dir, exist_ok=True)

# Generate config files for all combinations of seeds, models, torch_dtypes, and modes
for seed, feature_label_set, model, torch_dtype, mode in product(
    seeds, feature_label_sets, models, torch_dtypes, modes
):
    context_len = feature_label_set["context_length"]
    pred_len = feature_label_set["prediction_length"]

    # Set min_past to match context_length
    min_past = 60  # Fixed min_past value as per the new configuration

    # Get max_train_steps from mapping
    max_train_steps = max_train_steps_mapping.get(
        model, 200000
    )  # Default to 200k if model is missing

    # Define a unique folder for this configuration
    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{model.replace('/', '_')}_dtype_{torch_dtype}_mode_{mode}_context_{context_len}_pred_{pred_len}",
    )
    os.makedirs(config_folder, exist_ok=True)

    for patient_id in patients:
        # Create a patient-specific subfolder within the config group
        patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        # Define log directory inside the same patient folder
        log_folder = os.path.join(patient_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        # Define the dynamic data path using patient-specific path
        data_folder = f"/home/amma/LLM-TIME/data/standardized_arrow/{patient_id}-ws-training.arrow"
        test_data_folder = f"./data/standardized/{patient_id}-ws-testing.csv"

        # Prepare .gin configuration content
        config_content = f"""
run.log_dir = "./logs/"
run.chronos_dir = "/home/amma/"

run.data_settings = {{
    'path_to_train_data': '{data_folder}',
    'path_to_test_data': '{test_data_folder}',
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
    'tokenizer_kwargs': "{{'low_limit': -15,'high_limit': 15}}",
    'prediction_length': {pred_len},    
    'num_samples': 20,
    'context_length': {context_len},
    'min_past': {min_past},
    'learning_rate': 0.001,
    'max_train_steps': {max_train_steps},
    'save_steps': 100000,
    'log_steps': 500,
    'train_batch_size': 32,
    'random_init': False,
    'seed': {seed}  
}}
"""

        # Save .gin config file inside the patient-specific folder
        config_filename = "config.gin"
        config_path = os.path.join(patient_folder, config_filename)

        with open(config_path, "w") as f:
            f.write(config_content.strip())

        print(f"Generated: {config_path} with logs stored in {log_folder}")
