import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                logging.info('Updating learning rate to {}'.format(lr))
            else:
                logging.info('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                logging.info(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                logging.info(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)

def vali(llm_settings, accelerator, model, vali_loader, criterion, mae_metric):
    """
    Validation function to compute loss and MAE over the validation set.

    Args:
        llm_settings (dict): Settings for the LLM model, including prediction lengths and context lengths.
        accelerator (Accelerator): Accelerator object for distributed or mixed-precision training.
        model (torch.nn.Module): The LLM model being evaluated.
        vali_loader (DataLoader): DataLoader for the validation set.
        criterion (function): Loss function to evaluate the predictions.
        mae_metric (function): Metric function to calculate MAE.

    Returns:
        tuple: Average loss and MAE over the validation set.
    """
    total_loss = []
    total_mae_loss = []
    model.eval()

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader), desc="Validating"):
            # Move data to accelerator device
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -llm_settings['prediction_length']:, :]).float()
            dec_inp = torch.cat([batch_y[:, :llm_settings['context_length'], :], dec_inp], dim=1).to(accelerator.device)

            # Forward pass through the model
            if llm_settings.get('output_attention', False):
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Gather outputs and targets for metrics
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            # Extract relevant dimensions for univariate or multivariate settings
            f_dim = -1 if llm_settings['features'] == 'MS' else 0
            outputs = outputs[:, -llm_settings['prediction_length']:, f_dim:]
            batch_y = batch_y[:, -llm_settings['prediction_length']:, f_dim:]

            # Compute loss and metrics
            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    # Compute average loss and MAE
    avg_loss = np.mean(total_loss)
    avg_mae_loss = np.mean(total_mae_loss)

    # Switch model back to training mode
    model.train()

    return avg_loss, avg_mae_loss


def test(llm_settings, accelerator, model, train_loader, test_loader, criterion):
    """
    Test function to evaluate the model on the test dataset.

    Args:
        llm_settings (dict): Configuration settings for the model.
        accelerator (Accelerator): Accelerator object for distributed or mixed-precision training.
        model (torch.nn.Module): The trained model to be evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (function): Loss function to evaluate predictions.

    Returns:
        float: Average test loss over the entire test dataset.
    """
    # Retrieve the last insample window and test time series
    x, _ = train_loader.dataset.last_insample_window()
    y = test_loader.dataset.timeseries

    # Prepare inputs
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device).unsqueeze(-1)
    B, _, C = x.shape
    dec_inp = torch.zeros((B, llm_settings['prediction_length'], C)).float().to(accelerator.device)
    dec_inp = torch.cat([x[:, -llm_settings['context_length']:, :], dec_inp], dim=1)

    # Initialize outputs
    outputs = torch.zeros((B, llm_settings['prediction_length'], C)).float().to(accelerator.device)
    id_list = np.arange(0, B, llm_settings['prediction_batch_size'])
    id_list = np.append(id_list, B)

    model.eval()
    with torch.no_grad():
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )

        # Gather outputs and calculate loss
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if llm_settings['features'] == 'MS' else 0
        pred = outputs[:, -llm_settings['prediction_length']:, f_dim:]
        true = torch.tensor(np.array(y)).to(accelerator.device)

        loss = criterion(pred, true)

    model.train()
    return loss.item()



def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content

def load_txt_content(txt_path):
  
    with open(txt_path, 'r') as f:
        content = f.read()
    return content



def plot_predictions_with_overlays(
    timestamps,
    inputs,
    ground_truth,
    predictions,
    context_length,
    save_path=None,
    title="Prediction Overlays"
):
    """
    Plot predictions with overlays of context, ground truth, and predictions.
    
    :param timestamps: Array of timestamps for x-axis.
    :param inputs: Input data (context) used for predictions.
    :param ground_truth: Ground truth (actual values).
    :param predictions: Predicted values.
    :param context_length: Length of the context used for predictions.
    :param save_path: File path to save the plot (optional).
    :param title: Title of the plot.
    """
    num_plots = min(5, len(predictions))  # Limit to 5 plots for readability

    plt.figure(figsize=(12, 6))
    for idx in range(num_plots):
        context_end = context_length

        # Flatten or reshape inputs to ensure compatibility with timestamps
        input_context = inputs[idx].flatten()
        if len(input_context) != context_end:
            input_context = input_context[:context_end]

        plt.plot(
            timestamps[:context_end],
            input_context,
            label=f"Input Context {idx + 1}",
            linestyle='-',
            color="gray",
            alpha=0.7,
        )

        # Plot ground truth
        plt.plot(
            timestamps[context_end:context_end + len(ground_truth[idx])],
            ground_truth[idx].flatten(),
            label=f"Ground Truth {idx + 1}",
            linestyle='--',
            color="blue",
            alpha=0.8,
        )

        # Plot predictions
        plt.plot(
            timestamps[context_end:context_end + len(predictions[idx])],
            predictions[idx].flatten(),
            label=f"Prediction {idx + 1}",
            linestyle='-',
            color="red",
        )

    plt.title(title)
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend(loc="upper left", fontsize="small")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
    
def reformat_results(input_csv_path, output_csv_path):
    """
    Reformats the saved results CSV into a structured format with dynamic `t_i` columns for inputs,
    predictions, and ground truth values.

    Parameters:
    - input_csv_path: str
        Path to the input CSV file with grouped x_timestamps, y_timestamps, inputs, ground truth, and predictions.
    - output_csv_path: str
        Path to save the reformatted CSV file.
    """
    # Read the input CSV
    df = pd.read_csv(input_csv_path)

    # Initialize an empty list to hold reformatted rows
    reformatted_data = []

    for idx, row in df.iterrows():
        # Extract x_timestamps, y_timestamps, inputs, ground_truth, and predictions
        x_timestamps = row['x_timestamps'].split(', ')
        y_timestamps = row['y_timestamps'].split(', ')
        inputs = list(map(float, row['inputs'].split(', ')))
        ground_truth = list(map(float, row['ground_truth'].split(', ')))
        predictions = list(map(float, row['predictions'].split(', ')))

        # Calculate context and prediction lengths dynamically
        context_length = len(inputs)
        prediction_length = len(ground_truth)

        # Validate lengths
        assert len(x_timestamps) == context_length, (
            f"Mismatch in x_timestamps and inputs in row {idx}: "
            f"x_timestamps({len(x_timestamps)}), inputs({context_length})."
        )
        assert len(y_timestamps) == prediction_length, (
            f"Mismatch in y_timestamps and predictions/ground_truth in row {idx}: "
            f"y_timestamps({len(y_timestamps)}), ground_truth({prediction_length})."
        )

        # Create a row dictionary for this sequence
        reformatted_row = {}

        # Add x_timestamps and inputs
        for i in range(context_length):
            reformatted_row[f't_{i}_timestamp'] = x_timestamps[i]
            reformatted_row[f't_{i}_input'] = inputs[i]

        # Add y_timestamps, ground_truth, and predictions
        for i in range(prediction_length):
            pred_idx = context_length + i
            reformatted_row[f't_{pred_idx}_timestamp'] = y_timestamps[i]
            reformatted_row[f't_{pred_idx}_true'] = ground_truth[i]
            reformatted_row[f't_{pred_idx}_pred'] = predictions[i]

        # Append the reformatted row to the data
        reformatted_data.append(reformatted_row)

    # Convert the reformatted data to a DataFrame
    reformatted_df = pd.DataFrame(reformatted_data)

    # Save the reformatted DataFrame to a new CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    reformatted_df.to_csv(output_csv_path, index=False)
    print(f"Reformatted results saved to {output_csv_path}")
