import os
import glob
import json
from itertools import product
from seeds import fixed_seeds


def find_latest_checkpoint(training_folder_pattern, patient_id):
    """
    Finds the latest 'checkpoint-final' inside the correct patient folder.
    Example path:
    experiment_configs_chronos_training/seed_*_model_amazon_chronos-t5-*/patient_575/logs/logs_YYYY-MM-DD_HH-MM-SS/chronos-t5-*/run-0/checkpoint-final
    """
    print(
        f"\n🔍 Searching for checkpoint in: {training_folder_pattern} for patient {patient_id}"
    )

    try:
        # Expand wildcard pattern (match any seed)
        training_folders = glob.glob(training_folder_pattern)
        if not training_folders:
            print(
                f"⚠️ No matching training folders found for {training_folder_pattern}!"
            )
            return ""

        # Find all checkpoint-final files in the correct patient logs
        checkpoint_paths = []
        for folder in training_folders:
            patient_logs_path = os.path.join(folder, f"patient_{patient_id}", "logs")
            checkpoints = glob.glob(
                os.path.join(patient_logs_path, "**", "checkpoint-final"),
                recursive=True,
            )
            if checkpoints:
                checkpoint_paths.extend(checkpoints)

        if not checkpoint_paths:
            print(
                f"⚠️ No 'checkpoint-final' found for patient {patient_id} in {training_folder_pattern}!"
            )
            return ""

        # Sort checkpoints by modification time (latest first)
        checkpoint_paths.sort(key=os.path.getmtime, reverse=True)

        latest_checkpoint = checkpoint_paths[0]  # Pick the most recent one
        print(
            f"✅ Found latest checkpoint for patient {patient_id}: {latest_checkpoint}\n"
        )
        return latest_checkpoint

    except Exception as e:
        print(
            f"❌ Error finding checkpoint for patient {patient_id} in {training_folder_pattern}: {e}"
        )
        return ""


# Define parameter sets
feature_label_sets = [
    # {
    #     "input_features": [
    #         "BG_{t-5}",
    #         "BG_{t-4}",
    #         "BG_{t-3}",
    #         "BG_{t-2}",
    #         "BG_{t-1}",
    #         "BG_{t}",
    #     ],
    #     "labels": [
    #         "BG_{t+1}",
    #         "BG_{t+2}",
    #         "BG_{t+3}",
    #         "BG_{t+4}",
    #         "BG_{t+5}",
    #         "BG_{t+6}",
    #     ],
    #     "prediction_length": 6,
    #     "context_length": 6,
    # },
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

# Define parameters to iterate over
patients = [
    # "540",
    # "544",
    # "552",
    # "559",
    # "563",
    # "567",
    "570",
    # "575",
    "584",
    # "588",
    # "591",
    # "596",
]
seeds = fixed_seeds
models = ["amazon/chronos-t5-base",
        #   "amazon/chronos-t5-tiny"
          ]
torch_dtypes = ["float32"]
modes = ["inference"]

# Base directory for configurations
base_output_dir = "./experiment_configs_chronos_training_inference_noisy/"
os.makedirs(base_output_dir, exist_ok=True)

# Generate config files
for seed, feature_label_set, model, torch_dtype, mode in product(
    seeds, feature_label_sets, models, torch_dtypes, modes
):
    context_len = feature_label_set["context_length"]
    pred_len = feature_label_set["prediction_length"]
    min_past = context_len

    config_folder = os.path.join(
        base_output_dir,
        f"seed_{seed}_model_{model.replace('/', '-')}_dtype_{torch_dtype}_mode_{mode}_context_{context_len}_pred_{pred_len}",
    )
    os.makedirs(config_folder, exist_ok=True)

    for patient_id in patients:
        # Determine the correct model checkpoint path for this patient
        if "base" in model:
            training_folder_pattern = "./experiment_configs_chronos_training/seed_*_model_amazon_chronos-t5-base_dtype_*"
        else:
            training_folder_pattern = "./experiment_configs_chronos_training/seed_*_model_amazon_chronos-t5-tiny_dtype_*"

        checkpoint_path = find_latest_checkpoint(training_folder_pattern, patient_id)

        patient_folder = os.path.join(config_folder, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)
        log_folder = os.path.join(patient_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        data_folder = f"./data/noisy_formatted/{context_len}_{pred_len}"

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
    'restore_from_checkpoint': {bool(checkpoint_path)},
    'restore_checkpoint_path': '{checkpoint_path}',
    'seed': {seed}  
}}
"""

        config_filename = "config.gin"
        config_path = os.path.join(patient_folder, config_filename)

        with open(config_path, "w") as f:
            f.write(config_content.strip())

        print(f"✅ Generated: {config_path} with logs stored in {log_folder}")
