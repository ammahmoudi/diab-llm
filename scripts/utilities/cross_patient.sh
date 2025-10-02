python ./main.py --config_path ./configs/config_chronos_570_train_570_test.gin --log_level DEBUG
python ./main.py --config_path ./configs/config_chronos_570_train_584_test.gin --log_level DEBUG
python ./main.py --config_path ./configs/config_chronos_584_train_570_test.gin --log_level DEBUG
python ./main.py --config_path ./configs/config_chronos_584_train_584_test.gin --log_level DEBUG
./run_main.sh --config_path ./configs/config_time_llm_570_train_570_test.gin --log_level INFO
./run_main.sh --config_path ./configs/config_time_llm_570_train_584_test.gin --log_level DEBUG
./run_main.sh --config_path ./configs/config_time_llm_584_train_570_test.gin --log_level DEBUG
./run_main.sh --config_path ./configs/config_time_llm_584_train_584_test.gin --log_level DEBUG

