import logging
import numpy as np
import pandas as pd
from data_processing.data_scaler import DataScaler
from data_processing.data_sets import Dataset_T1DM, TimeSeriesDataset
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
                'preprocess_label': False,
                'percent': 100,
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

    def _log_dataset_info(self, dataframe, split, percent):
        """
        Logs the dataset size and split details.
        """
        logging.info(f"Dataset '{split.upper()}' initialized:")
        logging.info(f" - Total samples after applying {percent}%: {len(dataframe)}")
        logging.info(f" - Input features: {self._settings['input_features']}")
        logging.info(f" - Labels: {self._settings['labels']}")

    def load_from_csv(self, path_to_csv, split='train'):
        logging.info(f"Loading data from {path_to_csv} for split '{split}'.")

        dataframe = pd.read_csv(path_to_csv)
        total_rows = len(dataframe)

        # Apply percentage sampling
        percent = self._settings.get('percent', 100)
        if percent < 100:
            rows_to_include = int(total_rows * percent / 100)
            dataframe = dataframe.iloc[:rows_to_include]

        self._log_dataset_info(dataframe, split, percent)

        # Preprocess data if required
        if self._settings['preprocess_input_features'] or self._settings['preprocess_label']:
            if split == 'train':
                logging.info("Fitting scaler on train data.")
                self.scaler.fit(dataframe)

            logging.info("Applying scaler to the dataset.")
            dataframe = self.scaler.transform(dataframe)

        return dataframe

    def convert_csv_to_arrow(self, path_to_csv, split='train'):
        logging.info(f"Converting {path_to_csv} to Arrow format for split '{split}'.")

        dataframe = self.load_from_csv(path_to_csv, split=split)
        df_values = dataframe[self._settings['labels'] + self._settings['input_features']].values

        # Set an arbitrary start time
        start = np.datetime64("2000-01-01 00:00", "s")
        dataset = [
            {
                "start": start,
                "target": np.array(value, np.float32)
            } for value in df_values
        ]

        arrow_path = path_to_csv.replace('.csv', '.arrow')
        ArrowWriter(compression='lz4').write_to_file(dataset, arrow_path)
        logging.info(f"Arrow file saved at {arrow_path}.")

        return arrow_path

    def split_dataframe_input_features_vs_labels(self, dataframe):
        logging.info("Splitting dataframe into input features and labels.")

        input_features = dataframe[self._settings['input_features']]
        ground_truth = dataframe[self._settings['labels']]

        logging.info("Split completed:")
        logging.info(f" - Input features shape: {input_features.shape}")
        logging.info(f" - Labels shape: {ground_truth.shape}")

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
                'frequency':'5min',
                'num_workers': 1,
                'sequence_length': 6,
                'context_length': 6,
                'prediction_length': 6,
                'percent': 100,
                'val_split': 0  # Default value for val_split
            }
    ):
                       # TODO: Fix using our own data scaler.(currently if enabled it fits the standard scaler on train set)
        # self.scaler = DataScaler(
        #     settings={
        #         'input_features': settings['input_features'],
        #         'labels': settings['labels'],
        #         'method': settings['preprocessing_method'],
        #         'preprocess_input_features': settings['preprocess_input_features'],
        #         'preprocess_label': settings['preprocess_label']
        #     }
        # )
        
        self._settings = settings
        self.scaler = None
        self._settings = settings

    def load_from_csv(self, path_to_csv, batch_size=32, split='train'):
        """
        Load data from CSV and prepare DataLoader.
        Handles the case when validation data is not needed (val_split == 0).
        """
        if split == 'test':
            shuffle_flag = False
            drop_last = True
        else:
            shuffle_flag = True
            drop_last = True

        # Handle val_split == 0, return empty val_loader
        if self._settings['val_split'] == 0 and split == 'val':
            val_loader = DataLoader(
                [],
                batch_size=batch_size,
                shuffle=False,  # No shuffling since the loader is empty
                num_workers=self._settings['num_workers'],
                drop_last=False  # Avoid dropping last batch in an empty dataset
            )
            return pd.DataFrame(), val_loader
        # Initialize dataset
        data_set = Dataset_T1DM(
            data_path=path_to_csv,
            flag=split,
            size=(
                self._settings['sequence_length'],
                self._settings['context_length'],
                self._settings['prediction_length']
            ),
            features='S',
            target=self._settings['labels'][0],
            scale=self._settings['preprocess_input_features'],
            scaler=self.scaler,
            timeenc=0,
            freq=self._settings['frequency'],
            percent=self._settings['percent'],
            val_split=self._settings['val_split']  # Pass val_split to Dataset_T1DM
        )


        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self._settings['num_workers'],
            drop_last=drop_last
        )

        # Save the fitted scaler in train mode to use in other modes
        if split == 'train':
            self.scaler = data_set.scaler

        return data_set, data_loader
