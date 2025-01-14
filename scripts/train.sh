accelerate launch --num_processes=1 --num_machines=1 --dynamo_backend=no --mixed_precision bf16 main.py --config_path ./configs/config_time_llm.gin
