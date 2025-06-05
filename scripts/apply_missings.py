import os
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

    def save_to_csv(self, data: pd.DataFrame, filename):
        data.to_csv(filename, index=False)

def process_all_files_in_folder(input_folder, output_folder, miss_rate=0.1, missing_type='periodic'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # List all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)
            
            # Process each file
            data_processor = DataProcessor(input_filepath)
            data = data_processor.load_data()
            data_with_missingness = data_processor.apply_missingness(data, miss_rate, missing_type)
            data_processor.save_to_csv(data_with_missingness, output_filepath)
            print(f"Processed and saved: {filename}")

# Example usage
input_folder = "./data/d1namo_standardized"  # Folder containing the CSV files
output_folder = "./data/d1namo_missing_data_random"    # Folder to save the processed files
miss_rate = 0.05  # Missingness rate
missing_type = 'random'  # Type of missingness ('random' or 'periodic')

process_all_files_in_folder(input_folder, output_folder, miss_rate, missing_type)
