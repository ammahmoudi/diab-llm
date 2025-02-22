import logging
import os
import random
import torch
import time
import numpy as np
import pandas as pd
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim, nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from data_processing.refromat_results import reformat_results
from utils.result_saver import save_results_and_generate_plots
from utils.time_llm_utils import EarlyStopping, adjust_learning_rate, vali
from utils.file_utils import load_txt_content
from utils.timefeatures import decode_manual_time_features, decode_time_features

from llms.ts_llm import TimeSeriesLLM
from models import time_llm as TimeLLMModel


class TimeLLM(TimeSeriesLLM):
    def __init__(self, settings, data_settings, log_dir="./logs", name="time_llm"):
        os.environ["CURL_CA_BUNDLE"] = ""
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        
        # Set up the logger using the root logger
        self.logger = logging.getLogger(__name__)  # Use the module logger
        self.logger.info("Initializing TimeLLM model...")

       
        super(TimeLLM, self).__init__(name=name)
        
        self._llm_settings = settings
        self._data_settings = data_settings
        self._log_dir = log_dir
        
        # Initialize TimeLLM model
        self.llm_model = TimeLLMModel.Model(configs=self._llm_settings).float()

        # Accelerator setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config="./configs/ds_config_zero2.json"
        )
        self.accelerator = Accelerator(
            kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin
        )
        self.fitted_scaler = None  # Global variable to store the fitted scaler

        # Load content for prompts
        self._llm_settings["content"] = load_txt_content(
            self._data_settings["prompt_path"]
        )
                # Capture logs from the accelerator and other libraries like DeepSpeed
        self._setup_external_loggers()

    def _setup_external_loggers(self):
        """
        Ensures that logs from external libraries like DeepSpeed, Accelerate, etc.,
        are captured by the root logger.
        """
        # List of external libraries you want to capture logs from
        loggers_to_propagate = [
            "accelerate", "deepspeed", "transformers", "torch.distributed", "logging"
        ]
        
        for logger_name in loggers_to_propagate:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)  # Ensure they log at INFO level or higher
            logger.propagate = True  # Propagate logs to root logger

            # We don't need to add a new file handler as it's already handled by your logger file.
            # Just ensure propagation and log level
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    break
            else:
                # If the logger doesn't already have a file handler, we add the root handler
                for handler in logging.getLogger().handlers:
                    logger.addHandler(handler)
        
    def load_model(self, checkpoint_path, is_inference=False):
        """
        Load a pre-trained model checkpoint for inference.
        This function handles the case where the checkpoint has the "module." prefix or not, depending on the mode.

        :param checkpoint_path: Path to the saved model checkpoint.
        :param is_inference: Whether the model is being loaded in inference mode (True for inference, False for training).
        """
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            try:
                # Load model weights
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=self.accelerator.device,
                    weights_only=True,  # Ensure that only model weights are loaded
                )
                state_dict = checkpoint

                if is_inference:
                    # If in inference mode, check for missing 'module.' prefix and remove it
                    if any(key.startswith('module.') for key in state_dict.keys()):
                        logging.info("Removing 'module.' prefix for inference.")
                        state_dict = {key[7:]: value for key, value in state_dict.items()}  # Remove 'module.' prefix
                else:
                    # If in training mode (train+test), ensure 'module.' prefix is present
                    if not all(key.startswith('module.') for key in state_dict.keys()):
                        logging.info("Adding 'module.' prefix for training.")
                        state_dict = {f'module.{key}': value for key, value in state_dict.items()}  # Add 'module.' prefix

                # Load the state_dict into the model
                self.llm_model.load_state_dict(state_dict)
                
            except TypeError:
                # For older versions of PyTorch without weights_only, fallback
                self.llm_model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.accelerator.device)
                )

            # Move model to the correct device and dtype
            self.llm_model.to(self.accelerator.device)
            self.llm_model = self.llm_model.to(self._llm_settings['torch_dtype'])  # Align dtype if needed
            
            logging.info("Model checkpoint loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")




    def train(self, train_data, train_loader, val_loader=None):
        """
        Train the model with the given training data.
        Always saves the model at the end of training.
        Returns:
            checkpoint_path (str): Path to the saved checkpoint.
        """
        if hasattr(train_data, "scaler"):
            self.fitted_scaler = train_data.scaler

        self.logger.info("Starting Training...")
        train_steps = len(train_loader)

        # Initialize optimizer, scheduler, and early stopping
        model_optim = optim.Adam(
            self.llm_model.parameters(), lr=self._llm_settings["learning_rate"]
        )
        scheduler = self._initialize_scheduler(model_optim, train_steps)
        checkpoint_dir = os.path.join(self._log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            accelerator=self.accelerator,
            patience=self._llm_settings["patience"],
            verbose=True,
            save_mode=False,
        )

        # Loss functions
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # Prepare data loaders and model for accelerator
        train_loader, val_loader, self.llm_model, model_optim, scheduler = (
            self.accelerator.prepare(
                train_loader, val_loader, self.llm_model, model_optim, scheduler
            )
        )
        early_stopping_saved = False
        train_loss_l = []
        val_loss_l = []

        # Training loop
        for epoch in range(self._llm_settings["train_epochs"]):
            # Training step
            train_loss = self._run_epoch(
                train_loader, criterion, model_optim, scheduler, is_training=True
            )
            train_loss_l.append(train_loss)

            # Validation step (only if val_loader is not None and not empty)
            if val_loader is not None and len(val_loader) > 0:
                val_loss, val_mae_loss = vali(
                    self._llm_settings,
                    self.accelerator,
                    self.llm_model,
                    val_loader,
                    criterion,
                    mae_metric,
                )
                val_loss_l.append(val_loss)
                self.logger.info(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}"
                )
                early_stopping(val_loss, self.llm_model, checkpoint_dir)
            else:
                # If no validation data is available, log and continue without early stopping
                self.logger.warning(f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | No validation data available.")

            # Adjust learning rate
            self._adjust_scheduler(scheduler, epoch, model_optim)

            # Early stopping check
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered.")
                early_stopping_saved = True
                break

        if not early_stopping_saved:
            self.logger.warning("Early stopping did not trigger. Saving final model checkpoint.")

        # Final checkpoint save
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        self.logger.info(f"Saving final model checkpoint at {final_checkpoint_path}.")
        model_to_save = self.accelerator.unwrap_model(self.llm_model)
        torch.save(model_to_save.state_dict(), final_checkpoint_path)

        return final_checkpoint_path, train_loss_l, val_loss_l


    def predict(self, test_loader, output_dir):
        """
        Predict the output for the test data, save predictions to files,
        and generate plots for analysis.

        :param test_loader: DataLoader for test data.
        :param output_dir: Directory to save the predictions, formatted results, and plots.
        :return: Tuple of (predictions, targets) as numpy arrays for evaluation.
        """
        self.llm_model.eval()
        predictions, targets, inputs, input_timestamps, output_timestamps, attention_maps = (
            [],
            [],
            [],
            [],
            [],
            []
        )

        if len(test_loader) == 0:
            self.logger.info("Warning: The test loader is empty. No data to predict.")
            return None, None

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.accelerator.device)
                batch_y = batch_y.float().to(self.accelerator.device)

                # Prepare decoder input
                dec_inp = (
                    torch.zeros_like(
                        batch_y[:, -self._llm_settings["prediction_length"] :, :]
                    )
                    .float()
                    .to(self.accelerator.device)
                )
                dec_inp = torch.cat(
                    [batch_y[:, : self._llm_settings["context_length"], :], dec_inp],
                    dim=1,
                )

                outputs, attn_weights = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self._llm_settings["features"] == "MS" else 0
                outputs = outputs[:, -self._llm_settings["prediction_length"] :, f_dim:]

                predictions.append(outputs.cpu().numpy())
                # print(outputs.cpu().numpy().shape)
                targets.append(
                    batch_y[:, -self._llm_settings["prediction_length"] :, f_dim:]
                    .cpu()
                    .numpy()
                )
                inputs.append(batch_x.cpu().numpy())
                input_timestamps.append(batch_x_mark.cpu().numpy())
                output_timestamps.append(batch_y_mark.cpu().numpy())

                # attention_maps.append(attn_weights[-1].to(torch.float32).cpu().numpy())
                # print("=====================", (attn_weights[-1].to(torch.float32).cpu().numpy()).shape, len(attention_maps))
                # attn_array = attn_weights[-1].to(torch.float32).cpu().numpy()
                # print("Attention shape:", attn_array.shape)

        if not predictions:
            logging.error(
                "No predictions were made. Ensure the test loader contains data."
            )
            return None, None

        predictions = np.concatenate(predictions, axis=0)
        print(predictions.shape)
        targets = np.concatenate(targets, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        input_timestamps = np.concatenate(input_timestamps, axis=0)
        output_timestamps = np.concatenate(output_timestamps, axis=0)

        # Decode x timestamps
        decoded_x_timestamps = []
        if self._llm_settings["timeenc"] == 0:
            for batch in input_timestamps:
                decoded_x_timestamps.extend(
                    decode_manual_time_features(
                        batch, freq=self._data_settings["frequency"]
                    )
                )
        else:
            decoded_x_timestamps = decode_time_features(
                input_timestamps.T, freq=self._data_settings["frequency"]
            )

        # Decode y timestamps
        decoded_y_timestamps = []
        if self._llm_settings["timeenc"] == 0:
            for batch in output_timestamps:
                decoded_y_timestamps.extend(
                    decode_manual_time_features(
                        batch, freq=self._data_settings["frequency"]
                    )
                )
        else:
            decoded_y_timestamps = decode_time_features(
                output_timestamps.T, freq=self._data_settings["frequency"]
            )

        grouped_x_timestamps = [
            decoded_x_timestamps[
                i
                * self._llm_settings["context_length"] : (i + 1)
                * self._llm_settings["context_length"]
            ]
            for i in range(len(inputs))
        ]
        grouped_y_timestamps = [
            decoded_y_timestamps[
                i
                * self._llm_settings["prediction_length"] : (i + 1)
                * self._llm_settings["prediction_length"]
            ]
            for i in range(len(targets))
        ]

        grouped_x_timestamps = [
            ", ".join(
                ts.strftime("%d-%m-%Y %H:%M:%S") for ts in group if ts is not None
            )
            for group in grouped_x_timestamps
        ]
        grouped_y_timestamps = [
            ", ".join(
                ts.strftime("%d-%m-%Y %H:%M:%S") for ts in group if ts is not None
            )
            for group in grouped_y_timestamps
        ]

        save_results_and_generate_plots(
            output_dir,
            predictions,
            targets,
            inputs,
            grouped_x_timestamps,
            grouped_y_timestamps,
        )

        # np.save(os.path.join(output_dir, "attention_maps.npy"), np.concatenate(attention_maps, axis=0))
        # print("attentions saved.")

        return predictions, targets

    def evaluate(
        self, llm_prediction, ground_truth_data, metrics=["rmse", "mae", "mape"]
    ):
        """
        Evaluate the model using specified metrics.
        """
        return super().evaluate(llm_prediction, ground_truth_data, metrics)

    def _initialize_scheduler(self, optimizer, train_steps):
        """
        Initialize learning rate scheduler.
        """
        if self._llm_settings["lradj"] == "COS":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=20, eta_min=1e-8
            )
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self._llm_settings["pct_start"],
            epochs=self._llm_settings["train_epochs"],
            max_lr=self._llm_settings["learning_rate"],
        )

    def _adjust_scheduler(self, scheduler, epoch, optimizer):
        """
        Adjust learning rate scheduler based on the epoch or scheduler type.

        :param scheduler: The learning rate scheduler instance.
        :param epoch: Current epoch number.
        :param optimizer: The optimizer associated with the model.
        """
        if self._llm_settings["lradj"] != "TST":
            # Non-TST scheduler adjustment
            if self._llm_settings["lradj"] == "COS":
                scheduler.step()
                self.logger.info(
                    f"Epoch {epoch + 1} | Learning rate after epoch: {optimizer.param_groups[0]['lr']:.10f}"
                )
            else:
                if epoch == 0:
                    self._llm_settings["learning_rate"] = optimizer.param_groups[0][
                        "lr"
                    ]
                    self.logger.info(
                        f"Epoch {epoch + 1} | Initial learning rate: {optimizer.param_groups[0]['lr']:.10f}"
                    )
                adjust_learning_rate(
                    accelerator=self.accelerator,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    args=self._llm_settings,
                    printout=True,
                )
        else:
            # TST scheduler adjustment (log current learning rate)
            self.logger.info(
                f"Epoch {epoch + 1} | Learning rate (TST): {scheduler.get_last_lr()[0]:.10f}"
            )

    def _run_epoch(self, loader, criterion, optimizer, scheduler, is_training=True):
        """
        Run a single training/validation epoch with per-iteration learning rate adjustment.
        """
        self.llm_model.train() if is_training else self.llm_model.eval()
        epoch_loss = []

        iter_count = 0
        time_now = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(
            enumerate(loader)
        ):
            batch_x = batch_x.float().to(self.accelerator.device)
            batch_y = batch_y.float().to(self.accelerator.device)
            batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
            batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

            # Prepare decoder input
            dec_inp = (
                torch.zeros_like(
                    batch_y[:, -self._llm_settings["prediction_length"] :, :]
                )
                .float()
                .to(self.accelerator.device)
            )
            dec_inp = torch.cat(
                [batch_y[:, : self._llm_settings["context_length"], :], dec_inp], dim=1
            )

            if is_training:
                optimizer.zero_grad()

            outputs = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self._llm_settings["features"] == "MS" else 0
            outputs = outputs[:, -self._llm_settings["prediction_length"] :, f_dim:]
            batch_y = batch_y[:, -self._llm_settings["prediction_length"] :, f_dim:]
            loss = criterion(outputs, batch_y)

            if is_training:
                self.accelerator.backward(loss)
                optimizer.step()

                # Adjust learning rate per iteration for TST
                if self._llm_settings["lradj"] == "TST":
                    adjust_learning_rate(
                        self.accelerator,
                        optimizer,
                        scheduler,
                        iter_count + 1,
                        self._llm_settings,
                        printout=False,
                    )
                    self.logger.info(
                        f"Iteration {iter_count + 1} | Learning rate (TST): {scheduler.get_last_lr()[0]:.10f}"
                    )

            epoch_loss.append(loss.item())

            # Logging every 100 iterations
            iter_count += 1
            if (i + 1) % 100 == 0:
                self.logger.info(
                    f"Iteration {i + 1}, Loss: {loss.item():.7f}, Avg Loss: {np.mean(epoch_loss):.7f}"
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * (
                    (len(loader) - (i + 1))
                    + len(loader) * (self._llm_settings["train_epochs"] - 1)
                )
                self.logger.info(
                    f"\tSpeed: {speed:.4f}s/iter; Remaining time: {left_time:.4f}s"
                )
                iter_count = 0
                time_now = time.time()

        return np.mean(epoch_loss)
