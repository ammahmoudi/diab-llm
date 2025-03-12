import numpy as np
import pandas as pd
import random

class DataProcessor:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        data = pd.read_csv(self.filename)
        return data

    def apply_missingness(self, data: pd.DataFrame, miss_rate=0.1, missing_type='periodic'):
        targets = data['target'].values
        
        if missing_type == 'random':
            num_values = len(targets)
            num_missing = int(num_values * miss_rate)
            missing_indices = random.sample(range(num_values), num_missing)
            targets[missing_indices] = 0
        
        elif missing_type == 'periodic':
            window_size = 6  # Number of consecutive values to set to zero
            num_rows = len(targets)
            num_missing = int(num_rows * miss_rate)
            num_periods = max(1, num_missing // window_size)
            step = max(1, num_rows // num_periods)
            
            for start_idx in range(0, num_rows, step):
                if start_idx + window_size <= num_rows:
                    targets[start_idx:start_idx + window_size] = 0
        
        else:
            raise ValueError("Invalid missing type. Choose either 'random' or 'periodic'.")
        
        data['target'] = targets
        return data

    def save_to_csv(self, data: pd.DataFrame, filename="output.csv"):
        data.to_csv(filename, index=False)

# Example usage
filename = "./data/standardized/570-ws-training.csv"  # Replace with the actual filename
data_processor = DataProcessor(filename)
data = data_processor.load_data()
data_with_missingness = data_processor.apply_missingness(data, miss_rate=0.05, missing_type='random')
data_processor.save_to_csv(data_with_missingness, "./570_missing_random_train.csv")
