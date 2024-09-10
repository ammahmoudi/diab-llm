import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv_path", dest="csv_path", default=".")
argparser.add_argument("--save_path", dest="save_path", default=".")
argparser.add_argument("--input_window_size", dest="input_window_size", default=6)
argparser.add_argument("--prediction_window_size", dest="prediction_window_size", default=6)
argparser.add_argument(
    "--input_features",
    dest="input_features",
    nargs='+',
    type=str,
    default=["BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"]
)
argparser.add_argument(
    "--labels",
    dest="labels",
    nargs='+',
    type=str,
    default=["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"]
)
args = argparser.parse_args()

csv_path = args.csv_path
save_path = args.save_path
input_window_size = args.input_window_size
prediction_window_size = args.prediction_window_size
input_features = args.input_features
labels = args.labels

def reformat_t1dm_bg_data(
        parameters={
            'data_path': '.',
            'save_path': '.',
            'input_window_size': 6,
            'prediction_window_size': 6,
            'input_features': ["BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"],
            'labels': ["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"]
        }
):
    # load data
    data = pd.read_csv(parameters['data_path'])
    transformed_data = {
        feature: [] for feature in parameters['input_features'] + parameters['labels']
    }
    offset = parameters['input_window_size']
    for i in range((len(data) - parameters['prediction_window_size']) // parameters['input_window_size']):
        # input_data = data['_value'][i * parameters['input_window_size']: (i + 1) * parameters['input_window_size']]
        input_data = data['_value'][i: parameters['input_window_size'] + i]
        for feature, value in zip(parameters['input_features'], input_data):
            transformed_data[feature].append(value)
        # label_data = data['_value'][offset + (i * parameters['prediction_window_size']): offset + ((i + 1) * parameters['prediction_window_size'])]
        label_data = data['_value'][offset + i: offset + parameters['prediction_window_size'] + i]
        for label, value in zip(parameters['labels'], label_data):
            transformed_data[label].append(value)

    transformed_data = pd.DataFrame(transformed_data)
    transformed_data.to_csv(
        parameters['save_path'] + '/t1dm_bg_data_transformed.csv',
        index=False
    )

if __name__ == '__main__':
    # parse config file
    reformat_t1dm_bg_data(
        parameters={
            'data_path': csv_path,
            'save_path': save_path,
            'input_window_size': input_window_size,
            'prediction_window_size': prediction_window_size,
            'input_features': input_features,
            'labels': labels
        }
    )