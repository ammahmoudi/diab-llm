import os
import json
from itertools import product
import random
from seeds import fixed_seeds

# Define parameter sets that must be consistent
llm_model_sets = [
    # {"llm_model": "GPT2", "llm_dim": 768},
    # {"llm_model": "LLAMA", "llm_dim": 4096},
    {"llm_model": "BERT", "llm_dim": 768},
]

patients = ["001",
            "002", "003","004","005", "006", "007"
            ]

length_sets = [
    {"sequence_length": 6, "context_length": 6, "prediction_length": 6, "patch_len": 6},
    {"sequence_length": 6, "context_length": 6, "prediction_length": 9, "patch_len": 6},
]

train_epochs_set = [0,20]

seeds = fixed_seeds[:]


modes = ["training+inference"]
torch_dtypes = ["bfloat16"]
model_ids = ["test"]

base_output_dir_inference = "./d1namo_experiment_configs_time_llm_inference"
base_output_dir_training = "./d1namo_experiment_configs_time_llm_training"

os.makedirs(base_output_dir_inference, exist_ok=True)
os.makedirs(base_output_dir_training, exist_ok=True)

for seed, llm_model_set, length_set, torch_dtype, mode, model_id, train_epochs in product(
    seeds, llm_model_sets, length_sets, torch_dtypes, modes, model_ids, train_epochs_set
):
    llm_model = llm_model_set["llm_model"]
    llm_dim = llm_model_set["llm_dim"]
    
    sequence_len = length_set["sequence_length"]
    context_len = length_set["context_length"]
    pred_len = length_set["prediction_length"]
    patch_len = length_set["patch_len"]

    model_comment = f"d1namo_time_llm_{llm_model}_{llm_dim}_{sequence_len}_{context_len}_{pred_len}_{patch_len}"
    
    # Select appropriate base directory based on epochs
    base_output_dir = base_output_dir_inference if train_epochs == 0 else base_output_dir_training
    
    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{llm_model}_{llm_dim}_seq_{sequence_len}_context_{context_len}_pred_{pred_len}_patch_{patch_len}_epochs_{train_epochs}",
    )
    os.makedirs(config_folder, exist_ok=True)

    for patient_id in patients:
        patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)
        
        log_folder = os.path.join(patient_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)
        
        data_folder = "./data/d1namo_standardized"

        config_content = f"""
run.log_dir = "{log_folder}"
run.data_settings = {{
    'path_to_train_data': '{data_folder}/{patient_id}-ws-training.csv',
    'path_to_test_data': '{data_folder}/{patient_id}-ws-testing.csv',
    'input_features': ['target'],
    'labels': ['target'],
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
    'llm_layers': 8,
    'llm_dim': {llm_dim},
    'num_workers': 1,
    'torch_dtype': '{torch_dtype}',
    'model_id': '{model_id}',
    'sequence_length': {sequence_len},
    'context_length': {context_len},
    'prediction_length': {pred_len},
    'patch_len': {patch_len},
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.001,
    'train_epochs': {train_epochs},
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
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': {seed}
}}
"""

        config_filename = "config.gin"
        config_path = os.path.join(patient_folder, config_filename)

        with open(config_path, "w") as f:
            f.write(config_content.strip())

        print(f"Generated: {config_path} with logs stored in {log_folder}")
