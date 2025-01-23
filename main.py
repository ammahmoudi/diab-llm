import os
import gin
import torch
import datetime
from data_processing.data_loader import ChronosDataHandler, TimeLLMDataHandler
from absl import app, flags
from llms.chronos import ChronosLLM
from llms.time_llm import TimeLLM
from utils.logger import setup_logging
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', './configs/config_time_llm.gin', 'Path to config file.')
flags.DEFINE_string('log_level', 'INFO', 'Logging level: DEBUG, INFO, WARNING, ERROR, or FATAL.')


@gin.configurable
def run(
        data_settings={
            'path_to_train_data': 'path/to/train/data',
            'path_to_test_data': 'path/to/test/data',
            'input_features': ['feature1', 'feature2'],
            'label': ['label1'],
            'preprocessing_method': 'min_max',
            'preprocess_input_features': False,
            'preprocess_label': False,
            'percent':100
 
        },
        llm_settings={
            'mode': 'inference',
            'method': 'chronos',
            'model': 'amazon/chronos-t5-base',
            'torch_dtype': 'float32',
            'ntokens': 4096,
            'tokenizer_kwargs': "{'low_limit': 35,'high_limit': 500}",
            'prediction_length': 64,
            'num_samples': 100,
            'context_length': 64,
            'min_past': 64,
            'prediction_batch_size': 64,
            'prediction_use_auto_split': False,
            'max_train_steps': 10000,
            'train_batch_size': 32,
            'random_init': False,
            'save_steps': 1000,
            'eval_metrics': ['rmse', 'mae', 'mape'],
            'restore_from_checkpoint': False,
            'restore_checkpoint_path': 'path/to/checkpoint',
        },
        log_dir="./logs",
        chronos_dir=".",
        time_llm_dir="."
):
    # logging files
     # Parse the log level from the command-line arguments
    log_level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'FATAL': logging.FATAL
    }
    log_level = log_level_map.get(FLAGS.log_level.upper(), logging.INFO)

    # Set up logging
    log_dir = setup_logging(log_dir, log_level=log_level)
    logging.info("Logging initialized at directory: {}".format(log_dir))
    
    # convert torch_dtype in llm_settings
    if 'torch_dtype' in llm_settings:
        if llm_settings['torch_dtype'] == 'float32':
            dtype = torch.float32
        elif llm_settings['torch_dtype'] == 'bfloat16':
            dtype = torch.bfloat16
        elif llm_settings['torch_dtype'] == 'float16':
            dtype = torch.float16
        else:
            logging.info("Torch data type {} not supported.".format(llm_settings['torch_dtype']))
            raise NotImplementedError
        
        # Set globally
        torch.set_default_dtype(dtype)
            

    if llm_settings['method'] == 'chronos':
        # load data
        data_loader = ChronosDataHandler(
            settings={
                'input_features': data_settings['input_features'],
                'labels': data_settings['labels'],
                'preprocessing_method': data_settings['preprocessing_method'],
                'preprocess_input_features': data_settings['preprocess_input_features'],
                'preprocess_label': data_settings['preprocess_label'],
                'percent': data_settings['percent']
            }
        )
        logging.info('Running Chronos model.')
        llm = ChronosLLM(
            settings={
                'model': llm_settings['model'],
                'device_map': "cuda",
                'torch_dtype': llm_settings['torch_dtype']
            },
            log_dir=log_dir
        )
        if llm_settings['mode'] == 'inference':
            # check if train and test data are csv files
            if data_settings['path_to_train_data'].endswith('.csv'):
                logging.debug("Loading train data from {}.".format(data_settings['path_to_train_data']))
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings['path_to_train_data'],
                    split='train'
                )
                data_settings['path_to_train_data'] = train_data_path
            if data_settings['path_to_test_data'].endswith('.csv'):
                logging.debug("Loading test data from {}.".format(data_settings['path_to_test_data']))
                test_data = data_loader.load_from_csv(data_settings['path_to_test_data'])

            # prediction
            llm_prediction,targets = llm.predict(
                test_data=test_data,
                data_loader=data_loader,
                settings={
                    'prediction_length': llm_settings['prediction_length'],
                    'num_samples': llm_settings['num_samples'],
                    'batch_size': llm_settings['prediction_batch_size'],
                    'auto_split': llm_settings['prediction_use_auto_split']
                }
            )
            
            if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(llm_prediction, targets, llm_settings['eval_metrics'])
                    logging.info(f"Metric results: {metric_results}")

        elif llm_settings['mode'] == 'training':
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos-forecasting".format(chronos_dir)):
                root_dir = os.getcwd()
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.chdir("cd {}".format(chronos_dir))
                os.system("git clone https://github.com/amazon-science/chronos-forecasting.git")
                os.chdir(root_dir)
            
            if data_settings['path_to_train_data'].endswith('.csv'):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings['path_to_train_data'],
                    split='train'
                )
                data_settings['path_to_train_data'] = train_data_path
            elif data_settings['path_to_train_data'].endswith('.arrow'):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            llm.train(
                data_settings['path_to_train_data'],
                chronos_dir,
                settings={
                    'max_steps': llm_settings['max_train_steps'],
                    'random_init': llm_settings['random_init'],
                    'save_steps': llm_settings['save_steps'],
                    'batch_size': llm_settings['train_batch_size'],
                    'prediction_length': llm_settings['prediction_length'],
                    'context_length': llm_settings['context_length'],
                    'min_past': llm_settings['min_past'],
                    'ntokens': llm_settings['ntokens'],
                    'tokenizer_kwargs': llm_settings['tokenizer_kwargs']
                }
            )

        elif llm_settings['mode'] == 'training+inference':
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos-forecasting".format(chronos_dir)):
                root_dir = os.getcwd()
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.chdir(chronos_dir)
                os.system("git clone https://github.com/amazon-science/chronos-forecasting.git")
                os.chdir(root_dir)

            if data_settings['path_to_train_data'].endswith('.csv'):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings['path_to_train_data'],
                    split='train'
                )
                data_settings['path_to_train_data'] = train_data_path
            elif data_settings['path_to_train_data'].endswith('.arrow'):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            # training
            llm_ckpt_path = llm.train(
                data_settings['path_to_train_data'],
                chronos_dir,
                settings={
                    'max_steps': llm_settings['max_train_steps'],
                    'random_init': llm_settings['random_init'],
                    'save_steps': llm_settings['save_steps'],
                    'batch_size': llm_settings['train_batch_size'],
                    'prediction_length': llm_settings['prediction_length'],
                    'context_length': llm_settings['context_length'],
                    'min_past': llm_settings['min_past'],
                    'ntokens': llm_settings['ntokens'],
                    'tokenizer_kwargs': llm_settings['tokenizer_kwargs']
                }
            )
            # load checkpoint
            llm.load_model(llm_ckpt_path)
            # inference and evaluation
            if data_settings['path_to_test_data'].endswith('.csv'):
                test_data = data_loader.load_from_csv(data_settings['path_to_test_data'])


            llm_prediction,targets = llm.predict(
                test_data=test_data,
                data_loader=data_loader,
                settings={
                    'prediction_length': llm_settings['prediction_length'],
                    'num_samples': llm_settings['num_samples'],
                    'batch_size': llm_settings['prediction_batch_size'],
                    'auto_split': llm_settings['prediction_use_auto_split']
                }
            )
            if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(llm_prediction, targets, llm_settings['eval_metrics'])
                    logging.info(f"Metric results: {metric_results}")

    elif llm_settings['method'] == 'time_llm':
        # Initialize TimeLLM DataLoader
        data_loader = TimeLLMDataHandler(settings={
            'input_features': data_settings['input_features'],
            'labels': data_settings['labels'],
            'preprocessing_method': data_settings['preprocessing_method'],
            'preprocess_input_features': data_settings['preprocess_input_features'],
            'preprocess_label': data_settings['preprocess_label'],
            'frequency': data_settings['frequency'],
            'num_workers': llm_settings['num_workers'],
            'sequence_length': llm_settings['sequence_length'],
            'context_length': llm_settings['context_length'],
            'prediction_length': llm_settings['prediction_length'],
            'percent': data_settings['percent']
        })

        # Load datasets
        train_data, train_loader = data_loader.load_from_csv(data_settings['path_to_train_data'], batch_size=llm_settings['train_batch_size'], split='train')
        val_data, val_loader = data_loader.load_from_csv(data_settings['path_to_train_data'], batch_size=llm_settings['train_batch_size'], split='val')
        test_data, test_loader = data_loader.load_from_csv(data_settings['path_to_test_data'], batch_size=llm_settings['prediction_batch_size'], split='test')

        # Initialize TimeLLM
        llm = TimeLLM(
            name=llm_settings['model_comment'],
            settings=llm_settings,
            data_settings=data_settings,
            log_dir=log_dir
        )

        # Handle modes
        if llm_settings['mode'] == 'inference':
            if llm_settings['restore_from_checkpoint']:
                llm.load_model(llm_settings['restore_checkpoint_path'])

            if len(test_loader) == 0:
                logging.warning("Test loader is empty. No predictions will be made.")
            else:
                # Save prediction results
                
                llm_prediction, targets = llm.predict(test_loader, output_dir=log_dir)

                if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(llm_prediction, targets, llm_settings['eval_metrics'])
                    logging.info(f"Metric results: {metric_results}")




        elif llm_settings['mode'] == 'training':
            llm.train(train_data=train_data, train_loader=train_loader, val_loader=val_loader)

        elif llm_settings['mode'] == 'training+inference':
            # Train the model and retrieve checkpoint path
            checkpoint_path = llm.train(train_data=train_data, train_loader=train_loader, val_loader=val_loader)

            # Load the saved checkpoint for inference
            llm.load_model(checkpoint_path)

            if len(test_loader) == 0:
                logging.warning("Test loader is empty. No predictions will be made.")
            else:
                llm_prediction, targets = llm.predict(test_loader, output_dir=log_dir)
                if llm_prediction is not None:
                    # Evaluate metrics
                    metric_results = llm.evaluate(llm_prediction, targets, llm_settings['eval_metrics'])
                    logging.info(f"Metric results: {metric_results}")

        else:
            logging.error(f"Unsupported mode: {llm_settings['mode']}")
            raise NotImplementedError
    else:
        logging.error("Method {} not supported.".format(llm_settings['method']))
        raise NotImplementedError

    # Save Gin's operative config to a file
    txt_file = open(log_dir + "/gin_config.txt", "w+")
    txt_file.write(gin.operative_config_str())
    txt_file.close()

def main(args):
    # parse config file
    gin.parse_config_file(FLAGS.config_path)
    run()

if __name__ == '__main__':
    # pass config path to main function
    app.run(main)