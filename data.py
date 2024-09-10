import pandas as pd
from absl import logging

class DataHandler:
    def __init__(
            self,
            settings={
                'input_features': ['feature1', 'feature2'],
                'label': ['label1'],
                'preprocessing_method': 'min_max',
                'preprocess_input_features': False,
                'preprocess_label': False
            }
    ):
        self.scaler = DataScaler(
            settings={
                'input_features': settings['input_features'],
                'label': settings['label'],
                'method': settings['preprocessing_method'],
                'preprocess_input_features': settings['preprocess_input_features'],
                'preprocess_label': settings['preprocess_label']
            }
        )
        self._settings = settings

    def load_csv_as_dataframe(self, path_to_csv, training_data=False):
        dataframe = pd.read_csv(path_to_csv)
        if self._settings['preprocess_input_features'] or self._settings['preprocess_label']:
            if training_data:
                self.scaler.fit(dataframe)

            dataframe = self.scaler.scale(pd.read_csv(path_to_csv))

        return dataframe

    def convert_csv_to_arrow(self, path_to_csv, training_data=False):
        dataframe = self.load_csv_as_dataframe(path_to_csv, training_data=training_data)
        dataframe.to_feather(path_to_csv.replace('.csv', '.arrow'))

        return path_to_csv.replace('.csv', '.arrow')

    def split_dataframe_input_features_vs_labels(self, dataframe):
        input_features = dataframe[self._settings['input_features']]
        labels = dataframe[self._settings['label']]

        return input_features, labels


class DataScaler:
    def __init__(
            self,
            settings={
                'input_features': ['feature1', 'feature2'],
                'label': ['label1'],
                'method': 'min_max',
                'preprocess_input_features': True,
                'preprocess_label': False
            }
    ):
        self._settings = settings
        self._scaler_parameters = {}

    def fit(self, dataframe):
        if self._settings['method'] == 'min_max':
            if self._settings['preprocess_input_features']:
                for feature in self._settings['input_features']:
                    self._scaler_parameters[feature] = {
                        'min': dataframe[feature].min(),
                        'max': dataframe[feature].max()
                    }
            if self._settings['preprocess_label']:
                for label in self._settings['label']:
                    self._scaler_parameters[label] = {
                        'min': dataframe[label].min(),
                        'max': dataframe[label].max()
                    }
        elif self._settings['method'] == 'z_score':
            if self._settings['preprocess_input_features']:
                for feature in self._settings['input_features']:
                    self._scaler_parameters[feature] = {
                        'mean': dataframe[feature].mean(),
                        'std': dataframe[feature].std()
                    }
            if self._settings['preprocess_label']:
                for label in self._settings['label']:
                    self._scaler_parameters[label] = {
                        'mean': dataframe[label].mean(),
                        'std': dataframe[label].std()
                    }
        else:
            logging.info("Method {} not supported.".format(self._settings['method']))
            raise NotImplementedError

    def min_max_scale(self, dataframe):
        if self._settings['preprocess_input_features']:
            for feature in self._settings['input_features']:
                dataframe[feature] = (dataframe[feature] - self._scaler_parameters[feature]['min']) / (self._scaler_parameters[feature]['max'] - self._scaler_parameters[feature]['min'])
        if self._settings['preprocess_label']:
            for label in self._settings['label']:
                dataframe[label] = (dataframe[label] - self._scaler_parameters[label]['min']) / (self._scaler_parameters[label]['max'] - self._scaler_parameters[label]['min'])

        return dataframe

    def min_max_rescale(self, dataframe):
        if self._settings['preprocess_input_features']:
            for feature in self._settings['input_features']:
                dataframe[feature] = dataframe[feature] * (self._scaler_parameters[feature]['max'] - self._scaler_parameters[feature]['min']) + self._scaler_parameters[feature]['min']
        if self._settings['preprocess_label']:
            for label in self._settings['label']:
                dataframe[label] = dataframe[label] * (self._scaler_parameters[label]['max'] - self._scaler_parameters[label]['min']) + self._scaler_parameters[label]['min']

        return dataframe

    def z_score_scale(self, dataframe):
        if self._settings['preprocess_input_features']:
            for feature in self._settings['input_features']:
                dataframe[feature] = (dataframe[feature] - self._scaler_parameters[feature]['mean']) / self._scaler_parameters[feature]['std']
        if self._settings['preprocess_label']:
            for label in self._settings['label']:
                dataframe[label] = (dataframe[label] - self._scaler_parameters[label]['mean']) / self._scaler_parameters[label]['std']

        return dataframe

    def z_score_rescale(self, dataframe):
        if self._settings['preprocess_input_features']:
            for feature in self._settings['input_features']:
                dataframe[feature] = dataframe[feature] * self._scaler_parameters[feature]['std'] + self._scaler_parameters[feature]['mean']
        if self._settings['preprocess_label']:
            for label in self._settings['label']:
                dataframe[label] = dataframe[label] * self._scaler_parameters[label]['std'] + self._scaler_parameters[label]['mean']

        return dataframe

    def scale(self, dataframe):
        if self._settings['method'] == 'min_max':
            return self.min_max_scale(dataframe)
        elif self._settings['method'] == 'z_score':
            return self.z_score_scale(dataframe)

    def rescale(self, dataframe):
        if self._settings['method'] == 'min_max':
            return self.min_max_rescale(dataframe)
        elif self._settings['method'] == 'z_score':
            return self.z_score_rescale(dataframe)
        else:
            logging.info("Method {} not supported.".format(self._settings['method']))
            raise NotImplementedError

        return dataframe
