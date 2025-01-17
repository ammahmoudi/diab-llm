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
from utils.timefeatures import decode_manual_time_features, decode_time_features
from utils.tools import adjust_learning_rate, EarlyStopping, load_txt_content, reformat_results, vali, test
from utils.losses import smape_loss
from llms.ts_llm import TimeSeriesLLM
from models import time_llm as TimeLLMModel

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Set fixed random seed for reproducibility
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class TimeLLM(TimeSeriesLLM):
    def __init__(self, settings, data_settings, log_dir="./logs", name="time_llm"):
        super(TimeLLM, self).__init__(name=name)
        self._llm_settings = settings
        self._data_settings = data_settings
        self._log_dir = log_dir

        # Initialize TimeLLM model
        self.llm_model = TimeLLMModel.Model(configs=self._llm_settings).float()

        # Accelerator setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./configs/ds_config_zero2.json')
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
        self.fitted_scaler = None  # Global variable to store the fitted scaler

        # Load content for prompts
        self._llm_settings['content'] = load_txt_content(self._data_settings['prompt_path'])

    def load_model(self, checkpoint_path):
        """
        Load a pre-trained model checkpoint for inference.
        
        :param checkpoint_path: Path to the saved model checkpoint.
        """
        logging.info(f"Loading model checkpoint from {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            try:
                self.llm_model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.accelerator.device, weights_only=True)
                )
            except TypeError:
                # For older PyTorch versions without weights_only, fallback
                self.llm_model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.accelerator.device)
                )
            self.llm_model.to(self.accelerator.device)
            self.llm_model = self.llm_model.to(torch.bfloat16)  # Align dtype if needed

            self.llm_model.eval()
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
        if hasattr(train_data, 'scaler'):
            self.fitted_scaler = train_data.scaler

        logging.info("Starting Training...")
        train_steps = len(train_loader)

        # Initialize optimizer, scheduler, and early stopping
        model_optim = optim.Adam(self.llm_model.parameters(), lr=self._llm_settings['learning_rate'])
        scheduler = self._initialize_scheduler(model_optim, train_steps)
        checkpoint_dir = os.path.join(self._log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stopping = EarlyStopping(
            accelerator=self.accelerator,
            patience=self._llm_settings['patience'],
            verbose=True,
            save_mode=False
        )

        # Loss functions
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # Prepare data loaders and model for accelerator
        train_loader, val_loader, self.llm_model, model_optim, scheduler = self.accelerator.prepare(
            train_loader, val_loader, self.llm_model, model_optim, scheduler
        )
        early_stopping_saved = False
        # Training loop
        for epoch in range(self._llm_settings['train_epochs']):
            train_loss = self._run_epoch(train_loader, criterion, model_optim, scheduler, is_training=True)

            # Validation
            if val_loader is not None:
                val_loss, val_mae_loss = vali(
                    self._llm_settings, self.accelerator, self.llm_model, val_loader, criterion, mae_metric
                )
                logging.info(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}"
                )
                early_stopping(val_loss, self.llm_model, checkpoint_dir)

            # Adjust learning rate
            self._adjust_scheduler(scheduler, epoch)

            # Early stopping
            if early_stopping.early_stop:
                logging.info("Early stopping triggered.")
                early_stopping_saved = True
                break
            
        if not early_stopping_saved:
            logging.warning('Early stopping did not trigger. Saving final model checkpoint.')
        
        final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
        logging.info(f"Saving final model checkpoint at {final_checkpoint_path}.")
        model_to_save = self.accelerator.unwrap_model(self.llm_model)
        torch.save(model_to_save.state_dict(), final_checkpoint_path)
       

        return final_checkpoint_path


    def predict(self, test_loader, save_path=None, freq='5min'):
        """
        Predict the output for the test data, save predictions to a CSV file, 
        and ensure compatibility with evaluation functions.

        :param test_loader: DataLoader for test data.
        :param save_path: Path to save the predictions CSV file.
        :param freq: Frequency of the timestamps (e.g., '5min').
        :return: Tuple of (predictions, targets) as numpy arrays for evaluation.
        """

        self.llm_model.eval()
        predictions, targets, inputs, input_timestamps, output_timestamps = [], [], [], [], []

        if len(test_loader) == 0:
            logging.info("Warning: The test loader is empty. No data to predict.")
            return None, None

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.accelerator.device)
                batch_y = batch_y.float().to(self.accelerator.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self._llm_settings['prediction_length']:, :]).float().to(
                    self.accelerator.device
                )
                dec_inp = torch.cat([batch_y[:, :self._llm_settings['context_length'], :], dec_inp], dim=1)

                outputs = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self._llm_settings['features'] == 'MS' else 0
                outputs = outputs[:, -self._llm_settings['prediction_length']:, f_dim:]

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y[:, -self._llm_settings['prediction_length']:, f_dim:].cpu().numpy())
                inputs.append(batch_x.cpu().numpy())
                input_timestamps.append(batch_x_mark.cpu().numpy())  # Collect x timestamps
                output_timestamps.append(batch_y_mark.cpu().numpy())  # Collect y timestamps

        if not predictions:  # Handle case where no batches were processed
            logging.error("No predictions were made. Ensure the test loader contains data.")
            return None, None

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        inputs = np.concatenate(inputs, axis=0)
        input_timestamps = np.concatenate(input_timestamps, axis=0)  # Combine x timestamps
        output_timestamps = np.concatenate(output_timestamps, axis=0)  # Combine y timestamps

        # Decode x timestamps
        decoded_x_timestamps = []
        if self._llm_settings['timeenc'] == 0:
            for batch in input_timestamps:
                decoded_x_timestamps.extend(decode_manual_time_features(batch, freq=freq))
        else:
            decoded_x_timestamps = decode_time_features(input_timestamps.T, freq=freq)

        # Decode y timestamps
        decoded_y_timestamps = []
        if self._llm_settings['timeenc'] == 0:
            for batch in output_timestamps:
                decoded_y_timestamps.extend(decode_manual_time_features(batch, freq=freq))
        else:
            decoded_y_timestamps = decode_time_features(output_timestamps.T, freq=freq)

        # Group x timestamps batch-wise
        grouped_x_timestamps = [
            decoded_x_timestamps[i * self._llm_settings['context_length']:(i + 1) * self._llm_settings['context_length']]
            for i in range(len(inputs))
        ]

        # Group y timestamps batch-wise
        grouped_y_timestamps = [
            decoded_y_timestamps[i * self._llm_settings['prediction_length']:(i + 1) * self._llm_settings['prediction_length']]
            for i in range(len(targets))
        ]

        # Convert grouped timestamps to strings
        grouped_x_timestamps = [
            ", ".join(ts.strftime('%d-%m-%Y %H:%M:%S') for ts in group if ts is not None)
            for group in grouped_x_timestamps
        ]
        grouped_y_timestamps = [
            ", ".join(ts.strftime('%d-%m-%Y %H:%M:%S') for ts in group if ts is not None)
            for group in grouped_y_timestamps
        ]

        logging.debug(f"Predictions shape: {predictions.shape}")
        logging.debug(f"Ground Truth shape: {targets.shape}")
        logging.debug(f"Inputs shape: {inputs.shape}")
        logging.debug(f"X Timestamps count: {len(grouped_x_timestamps)}")
        logging.debug(f"Y Timestamps count: {len(grouped_y_timestamps)}")

        logging.info("Prediction complete.")
            
        # Save results to CSV
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            results = {
                'x_timestamps': grouped_x_timestamps,  # Use grouped x timestamps
                'y_timestamps': grouped_y_timestamps,  # Use grouped y timestamps
                'inputs': [", ".join(map(str, inputs[i].flatten())) for i in range(len(inputs))],
                'ground_truth': [", ".join(map(str, targets[i].flatten())) for i in range(len(targets))],
                'predictions': [", ".join(map(str, predictions[i].flatten())) for i in range(len(predictions))]
            }
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            reformat_results(save_path, output_csv_path=save_path.replace('.csv', '_reformatted.csv'))
            logging.info(f"Results saved to {save_path}")

        return predictions, targets




    def evaluate(self, llm_prediction, ground_truth_data, metrics=['rmse', 'mae', 'mape']):
        """
        Evaluate the model using specified metrics.
        """
        return super().evaluate(llm_prediction, ground_truth_data, metrics)

    def _initialize_scheduler(self, optimizer, train_steps):
        """
        Initialize learning rate scheduler.
        """
        if self._llm_settings['lradj'] == 'COS':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
        return lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=train_steps,
            pct_start=self._llm_settings['pct_start'],
            epochs=self._llm_settings['train_epochs'],
            max_lr=self._llm_settings['learning_rate']
        )

    def _adjust_scheduler(self, scheduler, epoch):
        """
        Adjust learning rate scheduler based on the epoch.
        """
        if self._llm_settings['lradj'] == 'TST':
            scheduler.step(epoch + 1)
        else:
            scheduler.step()

    def _run_epoch(self, loader, criterion, optimizer, scheduler, is_training=True):
        """
        Run a single training/validation epoch.
        """
        self.llm_model.train() if is_training else self.llm_model.eval()
        epoch_loss = []

        iter_count = 0
        time_now = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(loader)):
            batch_x = batch_x.float().to(self.accelerator.device)
            batch_y = batch_y.float().to(self.accelerator.device)
            batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
            batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self._llm_settings['prediction_length']:, :]).float().to(
                self.accelerator.device
            )
            dec_inp = torch.cat([batch_y[:, :self._llm_settings['context_length'], :], dec_inp], dim=1)

            if is_training:
                optimizer.zero_grad()

            outputs = self.llm_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self._llm_settings['features'] == 'MS' else 0
            outputs = outputs[:, -self._llm_settings['prediction_length']:, f_dim:]
            batch_y = batch_y[:, -self._llm_settings['prediction_length']:, f_dim:]
            loss = criterion(outputs, batch_y)

            if is_training:
                self.accelerator.backward(loss)
                optimizer.step()

            epoch_loss.append(loss.item())

            # Logging every 100 iterations
            iter_count += 1
            if (i + 1) % 100 == 0:
                logging.info(
                    f"Iteration {i + 1}, Loss: {loss.item():.7f}, Avg Loss: {np.mean(epoch_loss):.7f}"
                )
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((len(loader) - (i + 1)) + len(loader) * (self._llm_settings['train_epochs'] - 1))
                logging.info(f"\tSpeed: {speed:.4f}s/iter; Remaining time: {left_time:.4f}s")
                iter_count = 0
                time_now = time.time()

        return np.mean(epoch_loss)
