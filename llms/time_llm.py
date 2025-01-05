import torch
import time
import numpy as np
import pandas as pd
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim, nn
from torch.optim import lr_scheduler
from utils.tools import adjust_learning_rate, EarlyStopping, vali, test, load_content
from utils.losses import smape_loss
from llms.ts_llm import TimeSeriesLLM


class TimeLLM(TimeSeriesLLM):
    def __init__(self, settings, log_dir="./logs",
            name = "time_llm"
                 ):
        super(TimeLLM, self).__init__(name=name)
        self._llm_settings = settings
        self._log_dir = log_dir

        # Initialize TimeLLM model
        self.llm_model = TimeLLM.Model(configs=self._llm_settings).float()

        # Accelerator setup
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    def train(self, train_loader, val_loader=None, test_loader=None):
        """
        Train the model with the given training data.
        Optionally validate and test during training.
        """
        train_steps = len(train_loader)

        # Initialize optimizer, scheduler, and early stopping
        model_optim = optim.Adam(self.llm_model.parameters(), lr=self._llm_settings['learning_rate'])
        scheduler = self._initialize_scheduler(model_optim, train_steps)
        early_stopping = EarlyStopping(
            accelerator=self.accelerator, patience=self._llm_settings['patience'], verbose=True
        )

        # Loss functions
        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        # Prepare data loaders and model for accelerator
        train_loader, val_loader, test_loader, self.llm_model, model_optim, scheduler = self.accelerator.prepare(
            train_loader, val_loader, test_loader, self.llm_model, model_optim, scheduler
        )

        # Training loop
        for epoch in range(self._llm_settings['train_epochs']):
            train_loss = self._run_epoch(train_loader, criterion, model_optim, scheduler, is_training=True)

            # Validation
            if val_loader is not None:
                val_loss, val_mae_loss = vali(
                    self._llm_settings, self.accelerator, self.llm_model, None, val_loader, criterion, mae_metric
                )
                self.accelerator.print(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Val Loss: {val_loss:.7f}"
                )
                early_stopping(val_loss, self.llm_model, "checkpoints/")

            # Test at every epoch
            if test_loader is not None:
                test_loss, test_mae_loss = vali(
                    self._llm_settings, self.accelerator, self.llm_model, None, test_loader, criterion, mae_metric
                )
                self.accelerator.print(
                    f"Epoch {epoch + 1} | Test Loss: {test_loss:.7f} | Test MAE Loss: {test_mae_loss:.7f}"
                )

            # Break after testing when early stopping triggers
            if val_loader is not None and early_stopping.early_stop:
                self.accelerator.print("Early stopping triggered.")
                break

            # Adjust learning rate
            self._adjust_scheduler(scheduler, epoch)


    def predict(self, test_loader):
        """
        Predict the output for the test data.
        """
        self.llm_model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.accelerator.device)
                dec_inp = torch.zeros_like(batch_y[:, -self._llm_settings['prediction_length']:, :]).float().to(
                    self.accelerator.device
                )
                dec_inp = torch.cat([batch_y[:, :self._llm_settings['label_len'], :], dec_inp], dim=1)

                outputs = self.llm_model(batch_x, None, dec_inp, None)
                f_dim = -1 if self._llm_settings['features'] == 'MS' else 0
                outputs = outputs[:, -self._llm_settings['prediction_length']:, f_dim:]

                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        self.accelerator.print(f"Predictions shape: {predictions.shape}")
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

        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(self.accelerator.device)
            batch_y = batch_y.float().to(self.accelerator.device)
            batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
            batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

            # Decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self._llm_settings['prediction_length']:, :]).float().to(
                self.accelerator.device
            )
            dec_inp = torch.cat([batch_y[:, :self._llm_settings['label_len'], :], dec_inp], dim=1)

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

        return np.mean(epoch_loss)
