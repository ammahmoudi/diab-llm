import os
import gin
import torch
import datetime
from data import DataHandler
from absl import app, flags, logging
from llms.chronos_llm import ChronosLLM

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', './config.gin', 'Path to config file.')

@gin.configurable
def run(
        data_settings={
            'path_to_train_data': 'path/to/train/data',
            'path_to_test_data': 'path/to/test/data',
            'input_features': ['feature1', 'feature2'],
            'label': ['label1'],
            'preprocessing_method': 'min_max',
            'preprocess_input_features': False,
            'preprocess_label': False
        },
        llm_settings={
            'mode': 'inference',
            'method': 'chronos',
            'model': 'amazon/chronos-t5-base',
            'torch_dtype': 'float32',
            'llm_prediction_length': 64,
            'llm_num_samples': 100,
            'eval_metrics': ['rmse', 'mae', 'mape'],
            'restore_from_checkpoint': False,
            'restore_checkpoint_path': 'path/to/checkpoint',
        },
        log_dir="./logs",
        chronos_dir="."
):
    # logging files
    log_dir = log_dir + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logging file
    logging.get_absl_handler().use_absl_log_file(program_name="log", log_dir=log_dir)
    # convert torch_dtype in llm_settings
    if llm_settings['torch_dtype'] == 'float32':
        llm_settings['torch_dtype'] = torch.float32
    elif llm_settings['torch_dtype'] == 'bfloat16':
        llm_settings['torch_dtype'] = torch.bfloat16
    elif llm_settings['torch_dtype'] == 'float16':
        llm_settings['torch_dtype'] = torch.float16
    else:
        logging.info("Torch data type {} not supported.".format(llm_settings['torch_dtype']))
        raise NotImplementedError

    # load data
    data_loader = DataHandler(
        settings={
            'input_features': data_settings['input_features'],
            'label': data_settings['label'],
            'preprocessing_method': data_settings['preprocessing_method'],
            'preprocess_input_features': data_settings['preprocess_input_features'],
            'preprocess_label': data_settings['preprocess_label']
        }
    )

    if llm_settings['method'] == 'chronos':
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
                    training_data=True
                )
                data_settings['path_to_train_data'] = train_data_path
            if data_settings['path_to_test_data'].endswith('.csv'):
                logging.debug("Loading test data from {}.".format(data_settings['path_to_test_data']))
                test_data = data_loader.load_csv_as_dataframe(data_settings['path_to_test_data'])
                test_input_features, test_labels = data_loader.split_dataframe_input_features_vs_labels(test_data)

            # prediction
            llm_prediction = llm.predict(
                test_data=test_input_features,
                settings={
                    'prediction_length': llm_settings['llm_prediction_length'],
                    'num_samples': llm_settings['llm_num_samples'],
                }
            )
            # metric evaluation
            metric_results = llm.evaluate(
                llm_prediction=llm_prediction,
                ground_truth_data=test_labels,
                metrics=llm_settings['eval_metrics']
            )
            logging.info("Metric results: {}".format(metric_results))

        elif llm_settings['mode'] == 'training':
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos-forecasting".format(chronos_dir)):
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.system("cd {}".format(chronos_dir))
                os.system("git clone https://github.com/amazon-science/chronos-forecasting.git")
            
            if data_settings['path_to_train_data'].endswith('.csv'):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings['path_to_train_data'],
                    training_data=True
                )
                data_settings['path_to_train_data'] = train_data_path
            elif data_settings['path_to_train_data'].endswith('.arrow'):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            llm.train(data_settings['path_to_train_data'])

        elif llm_settings['mode'] == 'training+inference':
            # check if chronos-forecasting repo is already cloned
            if not os.path.exists("{}/chronos-forecasting".format(chronos_dir)):
                # clone repo from https://github.com/amazon-science/chronos-forecasting.git
                os.system("cd {}".format(chronos_dir))
                os.system("git clone https://github.com/amazon-science/chronos-forecasting.git")

            if data_settings['path_to_train_data'].endswith('.csv'):
                train_data_path = data_loader.convert_csv_to_arrow(
                    data_settings['path_to_train_data'],
                    training_data=True
                )
                data_settings['path_to_train_data'] = train_data_path
            elif data_settings['path_to_train_data'].endswith('.arrow'):
                pass
            else:
                logging.info("Data format not supported.")
                raise NotImplementedError

            # training
            llm_ckpt_path = llm.train(data_settings['path_to_train_data'])
            # load checkpoint
            llm.load_ckpt(llm_ckpt_path)
            # inference and evaluation
            if data_settings['path_to_test_data'].endswith('.csv'):
                test_data = data_loader.load_csv_as_dataframe(data_settings['path_to_test_data'])
                test_input_features, test_labels = data_loader.split_dataframe_input_features_vs_labels(test_data)

            llm_prediction = llm.predict(
                test_data=test_input_features,
                settings={
                    'prediction_length': llm_settings['llm_prediction_length'],
                    'num_samples': llm_settings['llm_num_samples'],
                }
            )
            metric_results = llm.evaluate(
                llm_prediction=llm_prediction,
                ground_truth_data=test_labels,
                metrics=llm_settings['eval_metrics']
            )
            logging.info("Metric results: {}".format(metric_results))


    else:
        logging.info("Method {} not supported.".format(llm_settings['method']))
        raise NotImplementedError

def main(args):
    # parse config file
    gin.parse_config_file(FLAGS.config_path)
    run()

if __name__ == '__main__':
    # pass config path to main function
    app.run(main)