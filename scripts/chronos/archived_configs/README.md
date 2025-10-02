# Archived Configuration Generators

This directory contains the original individual configuration generators that have been replaced by the unified `config_generator_chronos.py` in the parent directory.

## Archived Files:
- `config_generator_chronos_train.py` - Original training config generator
- `config_generator_chronos_inference.py` - Original inference config generator  
- `config_generator_chronos_trained_inference.py` - Original trained inference config generator
- `config_generator_chronos_trained_inference_lora.py` - Original LoRA inference config generator

## Migration Notes:
These files have been consolidated into the unified `../config_generator_chronos.py` which provides:
- All functionality from the original files
- Command-line interface with multiple modes
- Better maintainability and consistency
- Comprehensive examples in `../chronos_config_examples.sh`

## Usage:
Use the new unified generator instead:
```bash
# Training configs (replaces config_generator_chronos_train.py)
python3 ../config_generator_chronos.py --mode train

# Inference configs (replaces config_generator_chronos_inference.py)  
python3 ../config_generator_chronos.py --mode inference

# Trained inference configs (replaces config_generator_chronos_trained_inference.py)
python3 ../config_generator_chronos.py --mode trained_inference

# LoRA inference configs (replaces config_generator_chronos_trained_inference_lora.py)
python3 ../config_generator_chronos.py --mode lora_inference
```

These archived files are kept for reference and can be safely removed once you're confident the new unified generator works correctly for your use cases.