import pandas as pd

class DataLoader:
    def __init__(
            self,
            settings={
                'input_features': ['feature1', 'feature2'],
                'label': ['label1'],
            }
    ):
        self._settings = settings

    @staticmethod
    def load_csv_as_dataframe(path_to_csv):
        return pd.read_csv(path_to_csv)

    def convert_csv_to_arrow(self, path_to_csv):
        dataframe = self.load_csv_as_dataframe(path_to_csv)
        dataframe.to_feather(path_to_csv.replace('.csv', '.arrow'))

        return path_to_csv.replace('.csv', '.arrow')

    def split_dataframe_input_features_vs_labels(self, dataframe):
        input_features = dataframe[self._settings['input_features']]
        labels = dataframe[self._settings['label']]

        return input_features, labels
