import os
import warnings

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset

from utils.timefeatures import time_features

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
  def __init__(self, root_path, flag='train', size=None,
               features='S', data_path='ETTh1.csv',
               target='OT', scale=True, timeenc=0, freq='h'):
    # size [seq_len, label_len, pred_len]
    # info
    if size == None:
      self.seq_len = 24 * 4 * 4
      self.label_len = 24 * 4
      self.pred_len = 24 * 4
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self):
    self.scaler = MinMaxScaler()
    df_train = pd.read_csv(os.path.join(self.root_path,
                                      self.data_path))

    '''
    df_train.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_train.columns)
    cols.remove(self.target)
    cols.remove('date')
    df_train = df_train[['date'] + cols + [self.target]]
    # print(cols)
    num_train = int(len(df_train) * 0.7)
    num_test = int(len(df_train) * 0.2)
    num_vali = len(df_train) - num_train - num_test
    border1s = [0, num_train - self.seq_len, len(df_train) - num_test - self.seq_len]
    border2s = [num_train, num_train + num_vali, len(df_train)]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    if self.features == 'M' or self.features == 'MS':
      cols_data = df_train.columns[1:]
      df_data = df_train[cols_data]
    elif self.features == 'S':
      df_data = df_train[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0]:border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_train[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(['date'], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp


  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
  def __init__(self, root_path, flag='pred', size=None,
               features='S', data_path='ETTh1.csv',
               target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
    # size [seq_len, label_len, pred_len]
    # info
    if size == None:
      self.seq_len = 24 * 4 * 4
      self.label_len = 24 * 4
      self.pred_len = 24 * 4
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ['pred']

    self.features = features
    self.target = target
    self.scale = scale
    self.inverse = inverse
    self.timeenc = timeenc
    self.freq = freq
    self.cols = cols
    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self):
    self.scaler = MinMaxScaler()
    if type(self.data_path) == 'string':
      df_train = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
    else:
      df_train = self.data_path
    '''
    df_train.columns: ['date', ...(other features), target feature]
    '''
    if self.cols:
      cols = self.cols.copy()
      cols.remove(self.target)
    else:
      cols = list(df_train.columns)
      cols.remove(self.target)
      cols.remove('date')
    df_train = df_train[['date'] + cols + [self.target]]
    border1 = len(df_train) - self.seq_len
    border2 = len(df_train)

    if self.features == 'M' or self.features == 'MS':
      cols_data = df_train.columns[1:]
      df_data = df_train[cols_data]
    elif self.features == 'S':
      df_data = df_train[[self.target]]

    if self.scale:
      self.scaler.fit(df_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    tmp_stamp = df_train[['date']][border1:border2]
    tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
    pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

    df_stamp = pd.DataFrame(columns=['date'])
    df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
    if self.timeenc == 0:
      df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
      df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
      df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
      data_stamp = df_stamp.drop(['date'], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    if self.inverse:
      self.data_y = df_data.values[border1:border2]
    else:
      self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    if self.inverse:
      seq_y = self.data_x[r_begin:r_begin + self.label_len]
    else:
      seq_y = self.data_y[r_begin:r_begin + self.label_len]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self):
    return len(self.data_x) - self.seq_len + 1

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

class Dataset_Train(Dataset):
  def __init__(self, root_path, flag='train', size=None,
               features='S', data_path='ETTh1.csv',
               target='OT', scale=True, timeenc=0, freq='h'):
    # size [seq_len, label_len, pred_len]
    # info
    if size == None:
      self.seq_len = 24 * 4 * 4
      self.label_len = 24 * 4
      self.pred_len = 24 * 4
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ['train']
    type_map = {'train': 0}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self):
    self.scaler = MinMaxScaler()
    if type(self.data_path) == 'string':
      df_train = pd.read_csv(os.path.join(self.root_path, self.data_path))
    else:
      df_train = self.data_path

    '''
    df_train.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_train.columns)
    cols.remove(self.target)
    cols.remove('date')
    df_train = df_train[['date'] + cols + [self.target]]
    # print(cols)
    num_train = int(len(df_train) * 1)
    num_test = int(len(df_train) * 0)
    num_vali = len(df_train) - num_train - num_test
    border1s = [0, num_train - self.seq_len, len(df_train) - num_test - self.seq_len]
    border2s = [num_train, num_train + num_vali, len(df_train)]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    if self.features == 'M' or self.features == 'MS':
      cols_data = df_train.columns[1:]
      df_data = df_train[cols_data]
    elif self.features == 'S':
      df_data = df_train[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0]:border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_train[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(['date'], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp


  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

class Dataset_Battery(Dataset):
  def __init__(self, root_path=r'D:\Project\ETSformer\dataset\battery', flag='train', size=None,
               features='M', data_path='B0005_rev.csv',
               target='soh', scale=True, timeenc=0, freq='h', startpoint=0.8, vali=False):
    # size [seq_len, label_len, pred_len]
    # info
    if size == None:
      self.seq_len = 24 * 4 * 4
      self.label_len = 24 * 4
      self.pred_len = 24 * 4
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    self.vali = vali
    # if self.vali:
    #   assert flag in ['train', 'test', 'val']
    #   type_map = {'train': 0, 'val': 1, 'test': 2}
    # else:
    assert flag in ['train', 'test']
    type_map = {'train': 0, 'test': 1}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq
    self.startpoint = startpoint
    self.lag_window = 5

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self):
    self.scaler = MinMaxScaler()
    df_train = pd.read_csv(os.path.join(self.root_path,
                                      self.data_path))
    if 'date' not in df_train.columns:
      df_train = df_train.rename(columns={'datetime': 'date'})

    '''
    df_train.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_train.columns)
    cols.remove(self.target)
    cols.remove('date')
    only_cycle = False
    if only_cycle:
      df_train = df_train.drop_duplicates(subset='cycle', ignore_index=True)
    full = 1
    df_train = df_train[['date'] + cols + [self.target]]
    if len(df_train[df_train['capacity'] <= 1.4]) != 0:
      eol = df_train[df_train['capacity'] <= 1.4].index[0]
    else:
      eol = len(df_train)
    if full == 0:
      df_train = df_train.drop(index=range(eol, len(df_train)))
      cycle_len = df_train['cycle'].max()
    elif full == 1:
      cycle_len = df_train['cycle'].max()
    else:
      cycle_len = df_train['cycle'][eol]
    # Use Validation data or not
    if self.startpoint < 1:
      train_len = int(cycle_len * self.startpoint)
    else:
      train_len = self.startpoint
      # train_len = int(cycle_len - self.startpoint)
    # print(train_len)
    if self.vali:
      num_train = df_train[df_train["cycle"] == train_len].index[0]
      num_test = df_train[df_train["cycle"] == cycle_len - int(cycle_len * ((1 - self.startpoint) / 2))].index[0]
      num_test = len(df_train) - num_test
      num_vali = len(df_train) - num_train - num_test
      border1s = [0, num_train - self.seq_len, len(df_train) - num_test - self.seq_len]
      border2s = [num_train, num_train + num_vali, len(df_train)]
      border1 = border1s[self.set_type]
      border2 = border2s[self.set_type]
    else:
      num_train = df_train[df_train["cycle"] == train_len].index[0]
      num_test = len(df_train) - num_train
      border1s = [0, len(df_train) - num_test - self.seq_len]
      border2s = [num_train, len(df_train)]
      border1 = border1s[self.set_type]
      border2 = border2s[self.set_type]

    # cols.remove('cycle')
    # df_train = df_train.drop('cycle', axis=1)
    if self.features == 'M' or self.features == 'MS':
      cols_data = df_train.columns[1:]
      df_data = df_train[cols_data]
    elif self.features == 'S':
      df_data = df_train[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0]:border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_train[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(['date'], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

class Dataset_Battery_Combine(Dataset):
  def __init__(self, root_path=r'D:\Project\ETSformer\dataset\battery', flag='train', size=None,
               features='M', data_path='B0005_rev.csv',
               target='soh', scale=True, timeenc=0, freq='18s'):
    # size [seq_len, label_len, pred_len]
    # info
    if size == None:
      self.seq_len = 24 * 4 * 4
      self.label_len = 24 * 4
      self.pred_len = 24 * 4
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    # assert flag in ['train', 'test', 'val']
    # type_map = {'train': 0, 'val': 1, 'test': 2}
    assert flag in ['train', 'test']
    type_map = {'train': 0, 'test': 1}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self):
    self.scaler = MinMaxScaler()
    df_train = pd.read_csv(os.path.join(self.root_path,
                                        self.data_path))
    if self.data_path == 'CALCE_combine.csv':
      df_test = pd.read_csv(os.path.join(self.root_path, 'CS2_36.csv'))
    else:
      df_test = pd.read_csv(os.path.join(self.root_path, 'B0006_rev.csv'))
    if 'date' not in df_train.columns:
      df_train = df_train.rename(columns={'datetime': 'date'})
      df_test = df_test.rename(columns={'datetime': 'date'})
    df_combine = pd.concat([df_train, df_test], ignore_index=True)

    '''
    df_train.columns: ['date', ...(other features), target feature]
    '''
    cols = list(df_combine.columns)
    cols.remove(self.target)
    cols.remove('date')
    # print(df_combine.isnull().values.any())
    # df_combine = df_combine.drop('cycle', axis=1)

    num_train = len(df_train)
    num_test = len(df_combine) - num_train
    border1s = [0, len(df_combine) - num_test - self.seq_len]
    border2s = [num_train, len(df_combine)]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    # cols.remove('cycle')
    if self.features == 'M' or self.features == 'MS':
      cols_data = df_combine.columns[1:]
      df_data = df_combine[cols_data]
    elif self.features == 'S':
      df_data = df_combine[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0]:border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_combine[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(['date'], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)
