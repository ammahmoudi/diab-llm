import io
import logging
import os
import numpy as np
import yaml
import torch
from llms.ts_llm import TimeSeriesLLM
from chronos import ChronosPipeline

from utils.result_saver import save_results_and_generate_plots


class ChronosLLM(TimeSeriesLLM):
    def __init__(
        self,
        settings={
            "model": "amazon/chronos-t5-base",
            "device_map": "cuda",
            "torch_dtype": torch.float32,
        },
        log_dir="./logs",
        name="chronos_llm"
        """
        Initialize the Chronos LLM model.

        Args:
            settings (dict): Settings for the model, including model path, device map, and torch dtype.
            log_dir (str): Directory for logging outputs.
            name (str): Name of the LLM instance.
        """,
    ):
        super(ChronosLLM, self).__init__(name=name)
        self.llm_model = ChronosPipeline.from_pretrained(
            settings["model"],
            device_map=settings["device_map"],
            torch_dtype=settings["torch_dtype"],
        )
        self._llm_settings = settings
        self._log_dir = log_dir

    def load_model(self, ckpt_path):
        """
        Load a model checkpoint.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        logging.info(f"Loading model checkpoint from {ckpt_path}")
        self.llm_model = ChronosPipeline.from_pretrained(
            ckpt_path,
            device_map=self._llm_settings["device_map"],
            torch_dtype=self._llm_settings["torch_dtype"],
        )
        logging.info("Model checkpoint loaded successfully.")

    def predict(
        self,
        test_data,
        settings={
            'prediction_length': 64,
            'num_samples': 100,
            'batch_size': None,
            'auto_split': False
        }
    ):
        # batch dataframe into smaller sequences, e.g., always predict 64 samples (rows) at a time
        if settings['batch_size'] is not None:
            test_data_batches = [
                test_data[i:i + settings['batch_size']] for i in range(0, len(test_data), settings['batch_size'])
            ]
        else:
            test_data_batches = [test_data]

        prediction_results = []
        for batch in test_data_batches:
            if settings['prediction_length'] > 64 and settings['auto_split']:
                split_results = []
                # use a sliding window to predict the next 64 samples
                predictions_remaining = settings['prediction_length']
                num_splits = settings['prediction_length'] // 64
                num_splits = num_splits if settings['prediction_length'] % 64 == 0 else num_splits + 1
                current_llm_prediction = self.llm_model.predict(
                    context=torch.tensor(batch.values),
                    prediction_length=64,
                    num_samples=settings['num_samples'],
                    limit_prediction_length=True
                )
                if settings['num_samples'] > 1:
                    current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                split_results.append(torch.squeeze(current_llm_prediction))

                # update sliding window with current prediction
                batch.values[:, :(batch.values.shape[1] - 64)] = batch.values[:, 64:]
                batch.values[:, :64] = current_llm_prediction
                predictions_remaining -= 64
                num_splits -= 1

                for _ in range(num_splits):
                    current_llm_prediction = self.llm_model.predict(
                        context=torch.tensor(batch.values),
                        prediction_length=64 if predictions_remaining > 64 else predictions_remaining,
                        num_samples=settings['num_samples'],
                        limit_prediction_length=True
                    )
                    if settings['num_samples'] > 1:
                        current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                    split_results.append(torch.squeeze(current_llm_prediction))

                    # update sliding window with current prediction
                    if predictions_remaining > 64:
                        batch.values[:, :(batch.values.shape[1] - 64)] = batch.values[:, 64:]
                        batch.values[:, :64] = current_llm_prediction
                        predictions_remaining -= 64

                current_llm_prediction = torch.cat(split_results, dim=1)
            else:
                current_llm_prediction = self.llm_model.predict(
                    context=torch.tensor(batch.values),
                    prediction_length=settings['prediction_length'],
                    num_samples=settings['num_samples'],
                    limit_prediction_length=False
                )
                if settings['num_samples'] > 1:
                    current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                current_llm_prediction = torch.squeeze(current_llm_prediction)

            prediction_results.append(current_llm_prediction)

        # convert to tensor with shape (n_samples, n_features)
        llm_prediction = torch.cat(prediction_results, dim=0)
        llm_prediction = llm_prediction.cpu().detach().numpy()

        return llm_prediction








    def train(
            self,
            train_data_path,
            chronos_dir,
            settings={
                'max_steps': 10000,
                'random_init': True,
                'save_steps': 1000,
                'batch_size': 32,      
                'prediction_length': 64,
                'context_length': 64,
                'min_past': 64,
                'ntokens': 4096,
                'tokenizer_kwargs': "{'low_limit': 35,'high_limit': 500}"
            }
    ):
        with open("{}/chronos-forecasting/scripts/training/configs/{}.yaml".format(chronos_dir, self._llm_settings['model'].split("/")[-1]), 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        data_loaded['model_id'] = self._llm_settings['model']
        data_loaded['training_data_paths'] = [train_data_path]
        data_loaded['probability'] = [1.0]
        data_loaded['prediction_length'] = settings['prediction_length']
        data_loaded['context_length'] = settings['context_length']
        data_loaded['min_past'] = settings['min_past']
        data_loaded['tokenizer_kwargs'] = settings['tokenizer_kwargs']
        data_loaded['ntokens'] = settings['ntokens']
        data_loaded['random_init'] = settings['random_init']
        data_loaded['per_device_train_batch_size'] = settings['batch_size']
        data_loaded['max_steps'] = settings['max_steps']
        data_loaded['save_steps'] = settings['save_steps']

        output_dir = self._log_dir + "/{}".format(self._llm_settings['model'].split("/")[-1])
        data_loaded['output_dir'] = output_dir
        # store yaml file
        with io.open('{}/chronos-forecasting/scripts/training/configs/{}_custom.yaml'.format(chronos_dir, self._llm_settings['model'].split("/")[-1]), 'w',
                     encoding='utf8') as outfile:
            yaml.dump(data_loaded, outfile)

        os.system("python {}/chronos-forecasting/scripts/training/train.py --config {}/chronos-forecasting/scripts/training/configs/{}_custom.yaml".format(chronos_dir, chronos_dir, self._llm_settings['model'].split("/")[-1]))
        output_dir = "{}/run-0/checkpoint-final".format(output_dir)
        logging.info(output_dir)

        return output_dir