import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil

from tqdm import tqdm

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
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
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