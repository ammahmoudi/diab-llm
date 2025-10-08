#!/usr/bin/env python3
"""
Simple test to run inference on the successfully trained distilled model
"""

import os
import subprocess
from pathlib import Path

def create_simple_inference_config():
    """Create a simple inference config for the distilled model."""
    
    # Use the checkpoint path from the successful distillation
    checkpoint_path = "./distillation_experiments/full_pipeline_test/logs/distilled_tinybert_570/logs_2025-10-08_11-10-13/checkpoints/checkpoint.pth"
    
    config_content = f'''# Simple Inference Test for Distilled TinyBERT
# Dataset: 570

run.log_dir = "./distillation_experiments/full_pipeline_test/logs/inference_test/"

run.data_settings = {{
    'path_to_train_data': './data/ohiot1dm/raw_standardized/570-ws-training.csv',
    'path_to_test_data': './data/ohiot1dm/raw_standardized/570-ws-testing.csv',
    'input_features': ['target'],
    'labels': ['target'],
    'prompt_path': './data/ohiot1dm/raw_standardized/t1dm_prompt.txt',
    'preprocessing_method': 'min_max',
    'preprocess_input_features': False,
    'preprocess_label': False,
    'frequency': '5min',
    'percent': 100,
    'val_split': 0
}}

run.llm_settings = {{
    'task_name': 'long_term_forecast',
    'mode': 'inference',
    'method': 'time_llm',
    'llm_model': 'TinyBERT',
    'llm_layers': 4,
    'llm_dim': 312,
    'num_workers': 1,
    'torch_dtype': 'float32',
    'model_id': 'ohiot1dm_570',
    'sequence_length': 96,
    'context_length': 10,
    'prediction_length': 1,
    'patch_len': 16,
    'stride': 8,
    'prediction_batch_size': 64,
    'train_batch_size': 32,
    'learning_rate': 0.0005,
    'train_epochs': 1,
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
    'des': 'inference_test',
    'model_comment': 'distilled_tinybert_inference_test',
    'prompt_domain': 0,
    'timeenc': 0,
    'eval_metrics': ['rmse', 'mae', 'mape'],
    'seed': 238822,
    'restore_from_checkpoint': True,
    'restore_checkpoint_path': '{checkpoint_path}'
}}
'''
    
    return config_content

def main():
    base_dir = Path("/home/amma/LLM-TIME")
    configs_dir = base_dir / "configs" / "distillation_test"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧪 Testing Inference on Successfully Distilled Model")
    print("=" * 60)
    
    # Generate config
    config_content = create_simple_inference_config()
    config_path = configs_dir / "test_distilled_inference.gin"
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Simple inference config generated: {config_path}")
    
    # Run inference
    cmd = [
        "python", "main.py",
        "--config_path", str(config_path),
        "--log_level", "INFO"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    original_dir = os.getcwd()
    try:
        os.chdir(base_dir)
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully ran inference on distilled model!")
            return True
        else:
            print("❌ Inference test failed")
            return False
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()