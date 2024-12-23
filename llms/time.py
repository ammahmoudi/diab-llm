import torch
import time
import numpy as np
import pandas as pd
from llms.ts_llm import TimeSeriesLLM
from models.autoformer import Autoformer
from models.dlinear import Dlinear
from models.time_llm import TimeLLM
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim
from torch.optim import lr_scheduler
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, load_content, test
from utils.losses import smape_loss

# TODO: replace args with settings
class TimeLLM(TimeSeriesLLM):
    def __init__(self, settings):
        super(TimeLLM, self).__init__()
        if settings['model'] == 'Autoformer':
            model = Autoformer.Model(
                configs=settings['model_configs']
            ).float()
        elif settings['model'] == 'Dlinear':
            model = Dlinear.Model(
                configs=settings['model_configs']
            ).float()
        else:
            model = TimeLLM.Model(
                configs=settings['model_configs']
            ).float()

        self.model = model
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
        self.accelerator = accelerator

    def predict(
            self,
            train_loader,
            test_loader
    ):
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.accelerator.device)
        x = x.unsqueeze(-1)
        self.model.eval()

        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, args.pred_len, C)).float().to(self.accelerator.device)
            dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
            outputs = torch.zeros((B, args.pred_len, C)).float().to(self.accelerator.device)
            id_list = np.arange(0, B, args.eval_batch_size)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i]:id_list[i + 1], :, :] = self.model(
                    x[id_list[i]:id_list[i + 1]],
                    None,
                    dec_inp[id_list[i]:id_list[i + 1]],
                    None
                )
            self.accelerator.wait_for_everyone()
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

        self.accelerator.print('test shape:', preds.shape)
        if self.accelerator.is_local_main_process:
            forecasts_df = pd.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(args.pred_len)])
            forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
            forecasts_df.index.name = 'id'
            forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
            forecasts_df.to_csv(folder_path + args.seasonal_patterns + '_forecast.csv')

    def train(
            self,
            train_loader,
            test_loader,
    ):
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=args.patience, verbose=True)
        model_optim = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = smape_loss()

        train_loader, model, model_optim, scheduler = self.accelerator.prepare(
            train_loader, self.model, model_optim, scheduler)

        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.accelerator.device)

                batch_y = batch_y.float().to(self.accelerator.device)
                batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(self.accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                    self.accelerator.device)

                outputs = model(batch_x, None, dec_inp, None)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                batch_y_mark = batch_y_mark[:, -args.pred_len:, f_dim:]
                loss = criterion(batch_x, args.frequency_map, outputs, batch_y, batch_y_mark)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                self.accelerator.backward(loss)
                model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(self.accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            self.accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            test_loss = test(args, self.accelerator, model, train_loader, test_loader, criterion)
            vali_loss = test_loss
            self.accelerator.print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, model, path)  # model saving
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                adjust_learning_rate(self.accelerator, model_optim, scheduler, epoch + 1, args, printout=True)
            else:
                self.accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))