import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            path_to_csv,
            size,
            features,
            targets,
            flag='train',
            scaler=None,
            timeenc={'_ts': 0},
            freq='h',
            percent=100,
    ):
        assert len(size) == 3, "Size should be a iterable of 3 integers, e.g.,  [seq_len, label_len, pred_len]"
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['train', 'test', 'val'], "Flag should be either 'train', 'test' or 'val'"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.targets = targets
        self.scaler = scaler
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.path_to_csv = path_to_csv
        self.scaler = scaler
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        df_raw = pd.read_csv(self.path_to_csv)
        # TODO: substract sequence length
        border1s = [0, 0.7 * len(df_raw), 0.9 * len(df_raw)]
        border2s = [0.7 * len(df_raw), 0.9 * len(df_raw), len(df_raw)]

        border1 = int(border1s[self.set_type])
        border2 = int(border2s[self.set_type])

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        df_data = df_raw[self.features]
        df_data = df_data[border1s[0]:border2s[0]]

        if self.scaler is not None:
            if self.set_type == 0:
                self.scaler.fit(df_data)
            data = self.scaler.transform(df_raw)
        else:
            data = df_data

        df_stamp = df_raw[self.timeenc.keys()][border1:border2]
        df_stamp[self.timeenc.keys()] = pd.to_datetime(df_stamp[self.timeenc.keys()])
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
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)