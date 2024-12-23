import numpy as np
import pandas as pd
from data.data_scaler import DataScaler
from data.data_sets import TimeSeriesDataset
from torch.utils.data import DataLoader
from gluonts.dataset.arrow import ArrowWriter

class ChronosDataHandler:
    def __init__(
            self,
            settings={
                'input_features': ['feature1', 'feature2'],
                'labels': ['label1'],
                'preprocessing_method': 'min_max',
                'preprocess_input_features': False,
                'preprocess_label': False
            }
    ):
        self.scaler = DataScaler(
            settings={
                'input_features': settings['input_features'],
                'labels': settings['labels'],
                'method': settings['preprocessing_method'],
                'preprocess_input_features': settings['preprocess_input_features'],
                'preprocess_label': settings['preprocess_label']
            }
        )
        self._settings = settings

    def load_from_csv(self, path_to_csv, split='train'):
        dataframe = pd.read_csv(path_to_csv)
        if self._settings['preprocess_input_features'] or self._settings['preprocess_label']:
            if split == 'train':
                self.scaler.fit(dataframe)

            dataframe = self.scaler.transform(pd.read_csv(path_to_csv))

        return dataframe

    def convert_csv_to_arrow(self, path_to_csv, split='train'):
        # Note: Arrow dataset with two columns: start and target
        # start: start time of the time series
        # target: values of the time series - encoding: [historic values (=inputs), future values(=outputs)]
        dataframe = self.load_from_csv(path_to_csv, split=split)
        df_values = dataframe[self._settings['labels'] + self._settings['input_features']].values

        # Set an arbitrary start time
        start = np.datetime64("2000-01-01 00:00", "s")
        # always add ts interval to the start time
        dataset = [
            {
                "start": start,
                "target": np.array(value, np.float32)
            } for value in df_values
        ]

        ArrowWriter(compression='lz4').write_to_file(
            dataset,
            path=path_to_csv.replace('.csv', '.arrow'),
        )

        return path_to_csv.replace('.csv', '.arrow')

    def split_dataframe_input_features_vs_labels(self, dataframe):
        input_features = dataframe[self._settings['input_features']]
        ground_truth = dataframe[self._settings['labels']]

        return input_features, ground_truth


class TimeLLMDataHandler:
    def __init__(
            self,
            settings={
                'input_features': ['feature1', 'feature2'],
                'labels': ['label1'],
                'preprocessing_method': 'min_max',
                'preprocess_input_features': False,
                'preprocess_label': False,
                'num_workers': 1
            }
    ):
        self.scaler = DataScaler(
            settings={
                'input_features': settings['input_features'],
                'labels': settings['labels'],
                'method': settings['preprocessing_method'],
                'preprocess_input_features': settings['preprocess_input_features'],
                'preprocess_label': settings['preprocess_label']
            }
        )
        self._settings = settings

    def load_from_csv(self, path_to_csv, batch_size=32, split='train'):
        if split == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        data_set = TimeSeriesDataset(
            path_to_csv=path_to_csv,
            size=(
                self._settings['context_length'],
                self._settings['prediction_length'],
                self._settings['prediction_length']
            ),
            features=self._settings['input_features'],
            targets=self._settings['labels'],
            flag=split,
            scaler=self.scaler,
            timeenc=0,
            freq='h',
            percent=100
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self._settings['num_workers'],
            drop_last=drop_last
        )

        return data_set, data_loader
