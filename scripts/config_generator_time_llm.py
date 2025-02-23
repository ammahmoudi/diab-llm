import os
import json
from itertools import product
import random

# Define parameter sets that must be consistent
llm_model_sets = [
    {
        "llm_model": "GPT2",  # Example model
        "llm_dim": 768,       # Corresponding LLM dim
    },
    {
        "llm_model": "LLAMA", # Example model
        "llm_dim": 4096,      # Corresponding LLM dim
    },
    {
        "llm_model": "BERT", # Example model
        "llm_dim": 768,      # Corresponding LLM dim
    },
    # Add more models and dims as needed
]

# Define patients
patients = [
    "540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"
]

# Define sets for seq_len, context_len, pred_len, patch_len
length_sets = [
    {"sequence_length": 6, "context_length": 6, "prediction_length": 6, "patch_len": 6},
    {"sequence_length": 6, "context_length": 6, "prediction_length": 9, "patch_len": 6},
    # Add more combinations as needed
]

# Define train_epochs set
train_epochs_set = [0,20]  # Example number of training epochs

# Generate 10 random seeds between 0 and 999999 (or any range you like)
seeds = [random.randint(0, 999999) for _ in range(10)]
print(seeds)

# Define other static parameters
modes = ["training+inference"]
torch_dtypes = ["bfloat16"]
model_ids = ["test"]  # Example model IDs

# Base directory for configurations and logs
base_output_dir = "./experiment_configs_time_llm/"
os.makedirs(base_output_dir, exist_ok=True)

# Generate config files for all combinations
for seed, llm_model_set, length_set, torch_dtype, mode, model_id, train_epochs in product(
    seeds, llm_model_sets, length_sets, torch_dtypes, modes, model_ids, train_epochs_set
):
    llm_model = llm_model_set["llm_model"]
    llm_dim = llm_model_set["llm_dim"]
    
    # Parameters from length set
    sequence_len = length_set["sequence_length"]
    context_len = length_set["context_length"]
    pred_len = length_set["prediction_length"]
    patch_len = length_set["patch_len"]

    # Define the model comment and folder path
    model_comment = f"time_llm_{llm_model}_{llm_dim}_{sequence_len}_{context_len}_{pred_len}_{patch_len}"
    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{llm_model}_dim_{llm_dim}_seq_{sequence_len}_context_{context_len}_pred_{pred_len}_patch_{patch_len}_epochs_{train_epochs}",
    )
    os.makedirs(config_folder, exist_ok=True)

    for patient_id in patients:
        # Create a patient-specific subfolder
        patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        # Define log directory inside the same patient folder
        log_folder = os.path.join(patient_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        # Define the dynamic data path
        data_folder = f"./data/standardized"  # Adjust this to your actual data folder

        # Prepare .gin configuration content
        config_content = f"""
run.log_dir = "{log_folder}"
run.data_settings = {{
    'path_to_train_data': '{data_folder}/{patient_id}-ws-training.csv',
    'path_to_test_data': '{data_folder}/{patient_id}-ws-testing.csv',
    'input_features': ['_value'],
    'labels': ['_value'],
    'prompt_path': '{data_folder}/t1dm_prompt.txt',
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'frequency': '5min',
    'percent': 100,
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': '{mode}',
    'method': 'time_llm',
    'llm_model': '{llm_model}',
    'llm_layers': 32,
    'llm_dim': {llm_dim},  # 4096 for LLAMA, 768 for GPT2
    'num_workers': 1,
    'torch_dtype': '{torch_dtype}',
    'model_id': '{model_id}',
    'sequence_length': {sequence_len},
    'context_length': {context_len},
    'prediction_length': {pred_len},
    'patch_len': {patch_len},
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 1,
    'learning_rate': 0.001,
    'train_epochs': {train_epochs},  # Added train_epochs here
    'features': 'S',
    'd_model': 32,
    'd_ff': 32,
    'factor': 1,
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'e_layers': 2,
    'd_layers': 1,
    'n_heads': 8,
    'dropout': 0.1,
    'moving_avg': 25,
    'activation': 'gelu',
    'embed': 'timeF',
    'patience': 10,
    'lradj': 'COS',
    'des': 'test',
    'model_comment': '{model_comment}',
    'prompt_domain': 0,  # You can change this domain if needed
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {seed}
}}
"""

        # Save .gin config file inside the patient-specific folder
        config_filename = "config.gin"
        config_path = os.path.join(patient_folder, config_filename)

        with open(config_path, "w") as f:
            f.write(config_content.strip())

        print(f"Generated: {config_path} with logs stored in {log_folder}")
