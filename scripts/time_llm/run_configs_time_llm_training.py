import import_utils
import os
import subprocess
from utilities.extract_metrics import extract_metrics_to_csv

import sys
import os


# Base directory where config files are stored
base_output_dir = "./experiment_configs_time_llm_training/"
log_level = "INFO"

# Model priority order - updated to support all Time-LLM models
model_order = ["BERT", "DistilBERT", "TinyBERT", "BERT-tiny", "MiniLM", "MobileBERT", "ALBERT"
               # Large models for advanced training
               ,"GPT2", "OPT-125M"
               # Only include LLAMA for specific use cases due to size
               # ,"LLAMA"
               ]

# Recursively find all `config.gin` files and categorize by model type
config_files_by_model = {
    "GPT2": [], "BERT": [], "LLAMA": [], "DistilBERT": [], 
    "TinyBERT": [], "BERT-tiny": [], "MiniLM": [], 
    "MobileBERT": [], "ALBERT": [], "OPT-125M": []
}

for root, _, files in os.walk(base_output_dir):
    for file in files:
        if file == "config.gin":
            config_path = os.path.join(root, file)
            path_upper = config_path.upper()
            if "GPT2" in path_upper:
                config_files_by_model["GPT2"].append(config_path)
            elif "BERT-TINY" in path_upper or "BERT_TINY" in path_upper:
                config_files_by_model["BERT-tiny"].append(config_path)
            elif "TINYBERT" in path_upper:
                config_files_by_model["TinyBERT"].append(config_path)
            elif "DISTILBERT" in path_upper:
                config_files_by_model["DistilBERT"].append(config_path)
            elif "MOBILEBERT" in path_upper:
                config_files_by_model["MobileBERT"].append(config_path)
            elif "ALBERT" in path_upper:
                config_files_by_model["ALBERT"].append(config_path)
            elif "BERT" in path_upper:
                config_files_by_model["BERT"].append(config_path)
            elif "LLAMA" in path_upper:
                config_files_by_model["LLAMA"].append(config_path)
            elif "MINILM" in path_upper:
                config_files_by_model["MiniLM"].append(config_path)
            elif "OPT-125M" in path_upper or "OPT_125M" in path_upper:
                config_files_by_model["OPT-125M"].append(config_path)

# Run `run_main.sh` for each config in order of model priority
for model in model_order:
    for config_path in config_files_by_model[model]:
        command = f"./run_main.sh --config_path {config_path} --log_level {log_level} --remove_checkpoints True"
        print(f"Running: {command}")
        subprocess.run(command, shell=True)
        extract_metrics_to_csv(base_dir=base_output_dir, output_csv='./experiment_results_time_llm_training.csv')

# After running the main scripts, extract metrics
