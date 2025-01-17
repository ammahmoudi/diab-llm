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
            "prediction_length": 64,
            "num_samples": 100,
            "batch_size": None,
            "auto_split": False,
        },
    ):
        """
        Generate predictions using the model.

        Args:
            test_data (DataFrame): Test dataset.
            settings (dict): Prediction settings including batch size, prediction length, etc.

        Returns:
            tuple: Model predictions, ground truth targets, inputs, input timestamps, and output timestamps.
        """
        logging.info("Starting prediction...")

        if settings["batch_size"] is not None:
            test_data_batches = [
                test_data[i : i + settings["batch_size"]]
                for i in range(0, len(test_data), settings["batch_size"])
            ]
        else:
            test_data_batches = [test_data]

        prediction_results = []
        target_results = []
        input_timestamps = []
        output_timestamps = []
        inputs = []

        for batch in test_data_batches:
            batch_inputs = batch.values
            batch_input_timestamps = batch.index.values
            batch_output_timestamps = batch.index.values[
                -settings["prediction_length"] :
            ]

            if settings["prediction_length"] > 64 and settings["auto_split"]:
                split_results = []
                # use a sliding window to predict the next 64 samples
                predictions_remaining = settings["prediction_length"]
                num_splits = settings["prediction_length"] // 64
                num_splits = (
                    num_splits
                    if settings["prediction_length"] % 64 == 0
                    else num_splits + 1
                )
                current_llm_prediction = self.llm_model.predict(
                    context=torch.tensor(batch_inputs),
                    prediction_length=64,
                    num_samples=settings["num_samples"],
                    limit_prediction_length=True,
                )
                if settings["num_samples"] > 1:
                    current_llm_prediction = torch.mean(current_llm_prediction, dim=1)

                split_results.append(torch.squeeze(current_llm_prediction))

                # update sliding window with current prediction
                batch_inputs[:, : (batch_inputs.shape[1] - 64)] = batch_inputs[:, 64:]
                batch_inputs[:, :64] = current_llm_prediction
                predictions_remaining -= 64
                num_splits -= 1

                for _ in range(num_splits):
                    current_llm_prediction = self.llm_model.predict(
                        context=torch.tensor(batch_inputs),
                        prediction_length=(
                            64 if predictions_remaining > 64 else predictions_remaining
                        ),
                        num_samples=settings["num_samples"],
                        limit_prediction_length=True,
                    )
                    if settings["num_samples"] > 1:
                        current_llm_prediction = torch.mean(
                            current_llm_prediction, dim=1
                        )

                    split_results.append(torch.squeeze(current_llm_prediction))

                    # update sliding window with current prediction
                    if predictions_remaining > 64:
                        batch_inputs[:, : (batch_inputs.shape[1] - 64)] = batch_inputs[
                            :, 64:
                        ]
                        batch_inputs[:, :64] = current_llm_prediction
                        predictions_remaining -= 64

                current_llm_prediction = torch.cat(split_results, dim=1)
            else:
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
            target_results.append(batch_inputs)
            input_timestamps.append(batch_input_timestamps)
            output_timestamps.append(batch_output_timestamps)
            inputs.append(batch_inputs)

        # Combine results
        llm_prediction = torch.cat(prediction_results, dim=0).cpu().detach().numpy()
        targets = np.concatenate(target_results, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        grouped_x_timestamps = np.concatenate(input_timestamps, axis=0)
        grouped_y_timestamps = np.concatenate(output_timestamps, axis=0)

        # Save results and generate plots
        save_results_and_generate_plots(
            self._log_dir,
            llm_prediction,
            targets,
            inputs,
            grouped_x_timestamps,
            grouped_y_timestamps,
        )

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
        """
        Train the Chronos model.

        Args:
            train_data_path (str): Path to the training data.
            chronos_dir (str): Directory containing Chronos scripts and configs.
            settings (dict): Training settings including batch size, max steps, etc.

        Returns:
            str: Path to the final model checkpoint.
        """
        logging.info("Starting training...")

        config_file_path = f"{chronos_dir}/chronos-forecasting/scripts/training/configs/{self._llm_settings['model'].split('/')[-1]}.yaml"
        with open(config_file_path, "r") as stream:
            data_loaded = yaml.safe_load(stream)

        # Update configuration settings
        data_loaded.update(
            {
                "model_id": self._llm_settings["model"],
                "training_data_paths": [train_data_path],
                "prediction_length": settings["prediction_length"],
                "context_length": settings["context_length"],
                "min_past": settings["min_past"],
                "tokenizer_kwargs": settings["tokenizer_kwargs"],
                "ntokens": settings["ntokens"],
                "random_init": settings["random_init"],
                "per_device_train_batch_size": settings["batch_size"],
                "max_steps": settings["max_steps"],
                "save_steps": settings["save_steps"],
                "output_dir": self._log_dir,
            }
        )

        # Save the modified config to a custom YAML file
        custom_config_path = os.path.join(
            chronos_dir,
            f"chronos-forecasting/scripts/training/configs/{self._llm_settings['model'].split('/')[-1]}_custom.yaml",
        )
        with io.open(custom_config_path, "w", encoding="utf8") as outfile:
            yaml.dump(data_loaded, outfile)

        logging.info(f"Custom training configuration saved at {custom_config_path}")

        # Run the training script
        os.system(
            f"python {chronos_dir}/chronos-forecasting/scripts/training/train.py --config {custom_config_path}"
        )

        final_checkpoint_path = os.path.join(
            data_loaded["output_dir"], "run-0/checkpoint-final"
        )
        logging.info(
            f"Training completed. Final checkpoint saved at {final_checkpoint_path}"
        )

        return final_checkpoint_path
