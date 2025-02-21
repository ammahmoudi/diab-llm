import io
import logging
import os
import random
import numpy as np
import yaml
import torch
from llms.ts_llm import TimeSeriesLLM
from chronos import ChronosPipeline

from utils.result_saver import generate_run_title, save_results_and_generate_plots


class ChronosLLM(TimeSeriesLLM):
    def __init__(
        self,
        settings={
            "model": "amazon/chronos-t5-base",
            "device_map": "cuda",
            "torch_dtype": torch.float32,
        },
        data_settings={},
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
        self._llm_settings = settings
        self._data_settings=data_settings
        self._log_dir = log_dir

        # Get the existing root logger from setup_logging
        self.logger = logging.getLogger()
        self.logger.info("Initializing Chronos LLM model...")

        # Capture logs from ChronosPipeline
        chronos_logger = logging.getLogger("chronos")
        chronos_logger.setLevel(logging.DEBUG)
        chronos_logger.propagate = True  # Ensure it sends logs to the root logger

        # Load the Chronos pipeline with logging
        try:
            self.llm_model = ChronosPipeline.from_pretrained(
                 pretrained_model_name_or_path=settings["model"],
                 
                device_map=settings["device_map"],
                torch_dtype=settings["torch_dtype"],
                cache_dir=self._log_dir,
            )
            self.logger.info("Chronos LLM model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load Chronos LLM model: {e}")
            raise

    def load_model(self, ckpt_path):
        """
        Load a model checkpoint.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        self.logger.info(f"Loading model checkpoint from {ckpt_path}")
        self.llm_model = ChronosPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt_path,
            device_map=self._llm_settings["device_map"],
            torch_dtype=self._llm_settings["torch_dtype"],
            cache_dir=self._log_dir,
        )
        self.logger.info("Model checkpoint loaded successfully.")


    def predict(
        self,
        test_data,
        data_loader,
        settings={
            "prediction_length": 64,
            "num_samples": 100,
            "batch_size": None,
            "auto_split": False,
        },
    ):
        """
        Predict method to generate predictions and save inputs, targets, and predictions.

        Args:
            test_data: DataFrame containing both input features and targets.
            data_loader: Data loader of test data
            settings: Dictionary of settings for prediction.

        Returns:
            llm_prediction: Numpy array of model predictions.
        """
        
        logging.info("Starting prediction process...")

        # Split test data into input features and labels
        test_input_features, test_labels = data_loader.split_dataframe_input_features_vs_labels(test_data)
        logging.debug(f"Split test data into input features (shape: {test_input_features.shape}) and labels (shape: {test_labels.shape})")

        # Batch the data into smaller sequences if batch size is specified
        if settings["batch_size"] is not None:
            test_input_batches = [
                test_input_features[i : i + settings["batch_size"]]
                for i in range(0, len(test_input_features), settings["batch_size"])
            ]
            test_label_batches = [
                test_labels[i : i + settings["batch_size"]]
                for i in range(0, len(test_labels), settings["batch_size"])
            ]
            logging.info(f"Data is batched with batch size: {settings['batch_size']}")
        else:
            test_input_batches = [test_input_features]
            test_label_batches = [test_labels]
            logging.info("Processing data without batching.")

        prediction_results = []
        input_results = []
        target_results = []

        for batch_idx, (input_batch, target_batch) in enumerate(zip(test_input_batches, test_label_batches)):
            batch_inputs = input_batch.values
            batch_targets = target_batch.values

            input_results.append(batch_inputs)
            target_results.append(batch_targets)

            logging.debug(f"Processing batch {batch_idx+1}/{len(test_input_batches)} with input shape {batch_inputs.shape}")

            if settings["prediction_length"] > 64 and settings["auto_split"]:
                logging.info("Using auto-split mode for long sequence prediction.")
                split_results = []
                predictions_remaining = settings["prediction_length"]
                num_splits = settings["prediction_length"] // 64
                num_splits = num_splits if settings["prediction_length"] % 64 == 0 else num_splits + 1

                for split_idx in range(num_splits):
                    prediction_length = min(64, predictions_remaining)

                    logging.debug(f"Split {split_idx+1}/{num_splits}: Predicting {prediction_length} time steps.")

                    current_llm_prediction = self.llm_model.predict(
                        context=torch.tensor(batch_inputs),
                        prediction_length=prediction_length,
                        num_samples=settings["num_samples"],
                        limit_prediction_length=True,
                    )

                    if settings["num_samples"] > 1:
                        current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                    split_results.append(torch.squeeze(current_llm_prediction))

                    if predictions_remaining > 64:
                        batch_inputs[:, : (batch_inputs.shape[1] - 64)] = batch_inputs[:, 64:]
                        batch_inputs[:, :64] = current_llm_prediction
                        predictions_remaining -= 64
                        logging.debug(f"Sliding window updated: {predictions_remaining} steps remaining.")

                current_llm_prediction = torch.cat(split_results, dim=1)

            else:
                logging.info("Predicting in a single pass without auto-split.")
                current_llm_prediction = self.llm_model.predict(
                    context=torch.tensor(batch_inputs),
                    prediction_length=settings["prediction_length"],
                    num_samples=settings["num_samples"],
                    limit_prediction_length=False,
                )

                if settings["num_samples"] > 1:
                    current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                current_llm_prediction = torch.squeeze(current_llm_prediction)

            prediction_results.append(current_llm_prediction)
            logging.debug(f"Batch {batch_idx+1} prediction completed.")

        # Combine results
        llm_prediction = torch.cat(prediction_results, dim=0).cpu().detach().numpy()
        inputs = np.concatenate(input_results, axis=0)
        targets = np.concatenate(target_results, axis=0)

        logging.info(f"Prediction process completed. Shape of predictions: {llm_prediction.shape}")

        # Save the results
        short_name = generate_run_title(self._data_settings, self._llm_settings)
        save_results_and_generate_plots(
            self._log_dir,
            predictions=llm_prediction,
            targets=targets,
            inputs=inputs,
            name=short_name,
        )
        
        logging.info("Results saved successfully.")

        return llm_prediction, targets


    def train(
        self,
        train_data_path,
        chronos_dir,
        settings={
            "max_steps": 10000,
            "random_init": True,
            "save_steps": 1000,
            "batch_size": 32,
            "prediction_length": 64,
            "context_length": 64,
            "min_past": 64,
            "ntokens": 4096,
            "tokenizer_kwargs": "{'low_limit': 35,'high_limit': 500}",
        },
    ):
        with open(
            "{}/chronos-forecasting/scripts/training/configs/{}.yaml".format(
                chronos_dir, self._llm_settings["model"].split("/")[-1]
            ),
            "r",
        ) as stream:
            data_loaded = yaml.safe_load(stream)

        data_loaded["model_id"] = self._llm_settings["model"]
        data_loaded["training_data_paths"] = [train_data_path]
        data_loaded["probability"] = [1.0]
        data_loaded["prediction_length"] = settings["prediction_length"]
        data_loaded["context_length"] = settings["context_length"]
        data_loaded["min_past"] = settings["min_past"]
        data_loaded["tokenizer_kwargs"] = settings["tokenizer_kwargs"]
        data_loaded["ntokens"] = settings["ntokens"]
        data_loaded["random_init"] = settings["random_init"]
        data_loaded["per_device_train_batch_size"] = settings["batch_size"]
        data_loaded["max_steps"] = settings["max_steps"]
        data_loaded["save_steps"] = settings["save_steps"]
        if(self._llm_settings['seed']):
            data_loaded['seed']=self._llm_settings['seed']

        output_dir = self._log_dir + "/{}".format(
            self._llm_settings["model"].split("/")[-1]
        )
        data_loaded["output_dir"] = output_dir
        # store yaml file
        with io.open(
            "{}/chronos-forecasting/scripts/training/configs/{}_custom.yaml".format(
                chronos_dir, self._llm_settings["model"].split("/")[-1]
            ),
            "w",
            encoding="utf8",
        ) as outfile:
            yaml.dump(data_loaded, outfile)

        os.system(
            "python {}/chronos-forecasting/scripts/training/train.py --config {}/chronos-forecasting/scripts/training/configs/{}_custom.yaml".format(
                chronos_dir, chronos_dir, self._llm_settings["model"].split("/")[-1]
            )
        )
        output_dir = "{}/run-0/checkpoint-final".format(output_dir)
        self.logger.info(output_dir)

        return output_dir
