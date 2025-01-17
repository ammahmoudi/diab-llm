from data_processing.reformat_t1dm_bg_data import reformat_t1dm_bg_data
if __name__ == '__main__':
    # parse config file
    csv_path='/home/amma/LLM-TIME/data/raw/570-ws-training.csv'
    save_path='/home/amma/LLM-TIME/data/formated/570-ws-training.csv'
    input_window_size=6
    prediction_window_size=6
    input_features=["BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"]
    labels=["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"]
    
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
    csv_path='/home/amma/LLM-TIME/data/raw/570-ws-testing.csv'
    save_path='/home/amma/LLM-TIME/data/formated/570-ws-testing.csv'
    input_window_size=6
    prediction_window_size=6
    input_features=["BG_{t-5}", "BG_{t-4}", "BG_{t-3}", "BG_{t-2}", "BG_{t-1}", "BG_{t}"]
    labels=["BG_{t+1}", "BG_{t+2}", "BG_{t+3}", "BG_{t+4}", "BG_{t+5}", "BG_{t+6}"]
    
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