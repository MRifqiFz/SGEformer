import pandas as pd
from sklearn.model_selection import KFold
from models import ETSformer
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.Adam import Adam
from utils.constant import WRITER_PATH as wp

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
  def __init__(self, args):
    super(Exp_Main, self).__init__(args)

  def _build_model(self):
    model_dict = {
      'ETSformer': ETSformer,
    }
    model = model_dict[self.args.model](self.args).float()

    if self.args.use_multi_gpu and self.args.use_gpu:
      model = nn.DataParallel(model, device_ids=self.args.device_ids)
    return model

  def _get_data(self, flag):
    data_set, data_loader = data_provider(self.args, flag)
    return data_set, data_loader

  def _select_optimizer(self):
    if 'warmup' in self.args.lradj:
      lr = self.args.min_lr
    else:
      lr = self.args.learning_rate

    if self.args.smoothing_learning_rate > 0:
      smoothing_lr = self.args.smoothing_learning_rate
    else:
      smoothing_lr = 100 * self.args.learning_rate

    if self.args.damping_learning_rate > 0:
      damping_lr = self.args.damping_learning_rate
    else:
      damping_lr = 100 * self.args.learning_rate

    mlp_lr = self.args.mlp_lr

    nn_params = []
    smoothing_params = []
    damping_params = []
    mlp_params = []
    for k, v in self.model.named_parameters():
      if k[-len('_smoothing_weight'):] == '_smoothing_weight':
        smoothing_params.append(v)
      elif k[-len('_damping_factor'):] == '_damping_factor':
        damping_params.append(v)
      elif k[:len('mlp')] == 'mlp':
        mlp_params.append(v)
      else:
        nn_params.append(v)

    if self.args.optim == 'adam':
      model_optim = Adam([
        {'params': nn_params, 'lr': lr, 'name': 'nn'},
        {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
        {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
        # {'params': mlp_params, 'lr': mlp_lr, 'name': 'mlp'}
      ])
    elif self.args.optim == 'sgd':
      model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
    elif self.args.optim == 'rmsprop':
      model_optim = optim.RMSprop(self.model.parameters(), lr=self.args.learning_rate)

    return model_optim

  def _select_criterion(self):
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
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
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
    if self.args.val:
      vali_data, vali_loader = self._get_data(flag='val')
    test_data, test_loader = self._get_data(flag='test')
    writer = SummaryWriter(wp, comment=setting)

    path = os.path.join(self.args.checkpoints, setting)
    if not os.path.exists(path):
      os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    model_optim = self._select_optimizer()
    criterion = self._select_criterion()

    for epoch in range(self.args.train_epochs):
      iter_count = 0
      train_loss = []
      running_loss = 0

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
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        running_loss += loss.item()

        if (i + 1) % 100 == 0:
          print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
          speed = (time.time() - time_now) / iter_count
          left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
          print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
          writer.add_scalar('training loss', running_loss / 100, epoch * train_steps + i)
          iter_count = 0
          time_now = time.time()

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
        model_optim.step()

      print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

      list_trainLoss = train_loss
      train_loss = np.average(train_loss)
      if self.args.val:
        vali_loss = self.vali(vali_data, vali_loader, criterion)
      test_loss = self.vali(test_data, test_loader, criterion)

      if self.args.val:
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        early_stopping(vali_loss, self.model, path)
      else:
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, test_loss))
        early_stopping(train_loss, self.model, path)
      if early_stopping.early_stop:
        print("Early stopping")
        break

      adjust_learning_rate(model_optim, epoch + 1, self.args)

    best_model_path = path + '/' + r'checkpoint.pth'
    self.model.load_state_dict(torch.load(best_model_path))
    loss_list = np.array(list_trainLoss)
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    np.save(folder_path + 'train_loss.npy', loss_list)
    writer.add_hparams(
      {'model_id':self.args.model_id,
       'seq_len': self.args.seq_len,
       'pred_len': self.args.pred_len,
       'n_heads': self.args.n_heads,
       'd_model': self.args.d_model,
       'e_layers': self.args.e_layers,
       'd_layers': self.args.d_layers,
       'd_ff': self.args.d_ff,
       'activation': self.args.activation,
       'dropout': self.args.dropout,
       'K': self.args.K,
       'learning_rate': self.args.learning_rate,
       'lradj': self.args.lradj,
       'batch_size': self.args.batch_size,
       'startpoint': self.args.startpoint
       },
      {'hparam/train_loss':train_loss, 'hparam/test_loss': test_loss}
    )

    return self.model

  def train_cross(self, setting):
    dataset = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    writer = SummaryWriter(wp, comment=setting)

    path = os.path.join(self.args.checkpoints, setting)

    list_eval = []
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    if not os.path.exists(path):
      os.makedirs(path)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
      print("Fold Number:", fold)
      time_now = time.time()
      print(len(train_ids))
      print(len(test_ids))

      traindata, testdata = dataset.iloc[train_ids], dataset.iloc[test_ids]

      self.args.data_path = traindata
      train_data, train_loader = data_provider(self.args, 'train')
      self.args.data_path = testdata
      test_data, test_loader = data_provider(self.args, 'train')

      train_steps = len(train_loader)
      early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

      model_optim = self._select_optimizer()
      criterion = self._select_criterion()

      for epoch in range(self.args.train_epochs):
        iter_count = 0
        train_loss = []
        running_loss = 0

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
          outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

          f_dim = -1 if self.args.features == 'MS' else 0
          batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
          loss = criterion(outputs, batch_y)
          train_loss.append(loss.item())
          running_loss += loss.item()

          if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            writer.add_scalar('training loss', running_loss / 100, epoch * train_steps + i)
            iter_count = 0
            time_now = time.time()

          loss.backward()
          torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
          model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        test_loss = self.vali(test_data, test_loader, criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1, train_steps, train_loss, test_loss))
        early_stopping(test_loss, self.model, path)
        if early_stopping.early_stop:
          print("Early stopping")
          break

        adjust_learning_rate(model_optim, epoch + 1, self.args)

      preds = []
      trues = []

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
          outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

          f_dim = -1 if self.args.features == 'MS' else 0

          outputs = outputs[:, -self.args.pred_len:, f_dim:]
          batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
          outputs = outputs.detach().cpu().numpy()
          batch_y = batch_y.detach().cpu().numpy()

          pred = outputs
          true = batch_y

          preds.append(pred)
          trues.append(true)

      preds = np.array(preds)
      trues = np.array(trues)
      print('test shape:', preds.shape, trues.shape)
      preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
      trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
      print('test shape:', preds.shape, trues.shape)

      mae, mse, rmse, mape, mspe = metric(preds, trues)
      list_eval.append(mse)
      np.save(folder_path + f'metrics_f{fold}.npy', np.array([mae, mse, rmse, mape, mspe]))
      print(f'Fold {fold+1} MSE: {mse}')

    list_eval = np.array(list_eval)
    np.save(folder_path + 'list_mse.npy', list_eval)

    return

  def test(self, setting, data, save_vals=True):
    """data - 'val' or 'test' """
    test_data, test_loader = self._get_data(flag=data)
    writer = SummaryWriter(wp+'_test', comment=setting)

    print('loading model')
    self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    preds = []
    trues = []
    preds2 = []
    trues2 = []
    save_real = True

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
        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        if save_real:
          outputs2 = outputs[:, -self.args.pred_len:, 0:]
          batch_y2 = batch_y[:, -self.args.pred_len:, 0:].to(self.device)
          outputs2 = outputs2.detach().cpu().numpy()
          batch_y2 = batch_y2.detach().cpu().numpy()

          pred2 = outputs2
          true2 = batch_y2

          preds2.append(pred2)
          trues2.append(true2)

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()

        pred = outputs
        true = batch_y

        preds.append(pred)
        trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    if save_real:
      preds2 = np.array(preds2)
      trues2 = np.array(trues2)
      preds2 = preds2.reshape(-1, preds2.shape[-2], preds2.shape[-1])
      trues2 = trues2.reshape(-1, trues2.shape[-2], trues2.shape[-1])

    # result save
    folder_path = './results/' + setting + '/'
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
    writer.add_scalar('test mse', mse)
    writer.add_hparams(
      {'model_id':self.args.model_id,
       'seq_len': self.args.seq_len,
       'pred_len': self.args.pred_len,
       'n_heads': self.args.n_heads,
       'd_model': self.args.d_model,
       'e_layers': self.args.e_layers,
       'd_layers': self.args.d_layers,
       'd_ff': self.args.d_ff,
       'activation': self.args.activation,
       'dropout': self.args.dropout,
       'K': self.args.K,
       'learning_rate': self.args.learning_rate,
       'lradj': self.args.lradj,
       'batch_size': self.args.batch_size,
       'startpoint': self.args.startpoint
       },
      {'hparam/mae':mae, 'hparam/mse': mse, 'hparam/rmse': rmse}
    )

    np.save(folder_path + f'{data}_metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

    if save_vals:
      np.save(folder_path + 'pred.npy', preds)
      np.save(folder_path + 'true.npy', trues)
      if save_real:
        np.save(folder_path + 'pred2.npy', preds2)
        np.save(folder_path + 'true2.npy', trues2)

    return
