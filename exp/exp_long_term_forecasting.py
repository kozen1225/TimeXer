from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from statsmodels.tsa.seasonal import STL
from dtaidistance import dtw
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss.lower() == 'huber':
            criterion = nn.HuberLoss(delta=self.args.loss_delta)
        else: # 默认使用 MSE
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            print(f"Number of attention layers: {len(attention_list)}")
                            for i, att in enumerate(attention_list):
                                if att is not None:
                                    print(f"Attention from layer {i+1} shape: {att.shape}")
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        print(f"Number of attention layers: {len(attention_list)}")
                        for i, att in enumerate(attention_list):
                            if att is not None:
                                print(f"Attention from layer {i+1} shape: {att.shape}")
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            print(f"Number of attention layers: {len(attention_list)}")
                            for i, att in enumerate(attention_list):
                                if att is not None:
                                    print(f"Attention from layer {i+1} shape: {att.shape}")
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        print(f"Number of attention layers: {len(attention_list)}")
                        for i, att in enumerate(attention_list):
                            if att is not None:
                                print(f"Attention from layer {i+1} shape: {att.shape}")
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            print(f"Number of attention layers: {len(attention_list)}")
                            for i, att in enumerate(attention_list):
                                if att is not None:
                                    print(f"Attention from layer {i+1} shape: {att.shape}")
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, attention_list = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        print(f"Number of attention layers: {len(attention_list)}")
                        for i, att in enumerate(attention_list):
                            if att is not None:
                                print(f"Attention from layer {i+1} shape: {att.shape}")

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if self.args.features == 'MS':
                        outputs = np.tile(outputs, [1, 1, batch_y.shape[-1]])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # `gt` (ground truth) 包含了历史数据和真实的未来数据。
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # `pd` (prediction) 为了在图上清晰地展示，应该只包含预测的未来数据。
                    # 我们用 NaN 填充历史数据部分，这样绘图时就会跳过这部分。
                    pd = np.concatenate((np.full(input.shape[1], np.nan), pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # STL, DTW and Plotting
        try:
            # Take the first sample and first feature for analysis and plotting
            y_true_sample = trues[0, :, 0]
            y_pred_sample = preds[0, :, 0]

            # 1. STL Decomposition
            # STL requires the series to be long enough and have a period.
            # We choose a period, ensuring it's an odd integer and smaller than the series length.
            if len(y_true_sample) >= 14:
                period = 7
            else:
                period = max(3, int(len(y_true_sample) / 2) * 2 - 1)

            if len(y_true_sample) > 2 * period:
                stl = STL(y_true_sample, period=period, robust=True)
                res = stl.fit()
                trend_baseline = res.trend

                # 2. Calculate MAE and DTW
                mae_sample = np.mean(np.abs(y_true_sample - y_pred_sample))
                dtw_distance, _ = dtw.warping_paths(y_true_sample, y_pred_sample)

                print(f'--- Analysis for the first sample ---')
                print(f'MAE: {mae_sample:.4f}')
                print(f'DTW Distance: {dtw_distance:.4f}')

                # 3. Plotting
                plt.figure(figsize=(15, 7))
                plt.plot(y_true_sample, label='Original Ground Truth')
                plt.plot(trend_baseline, label='Trend Baseline (STL)')
                plt.plot(y_pred_sample, label='Prediction')
                plt.legend()
                plt.title(f'Forecast vs. Ground Truth vs. Trend ({setting})')
                
                # The folder_path is defined as './test_results/' + setting + '/' earlier
                # Let's use that one.
                plot_save_path = os.path.join(folder_path, 'trend_comparison_plot.png')
                plt.savefig(plot_save_path)
                plt.close()
                print(f'Comparison plot saved to {plot_save_path}')
            else:
                print("Time series is too short for STL decomposition.")

        except Exception as e:
            print(f"An error occurred during STL/DTW/Plotting: {e}")


        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, mae_std, mse_std = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, mse_std:{:.4f}, mae_std:{:.4f}'.format(mse, mae, mse_std, mae_std))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{:.4f}, mae:{:.4f}, mse_std:{:.4f}, mae_std:{:.4f}'.format(mse, mae, mse_std, mae_std))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, mae_std, mse_std]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
