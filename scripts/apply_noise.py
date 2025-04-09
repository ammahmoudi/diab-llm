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

    def apply_gaussian_noise(self, data: pd.DataFrame, noise_std=0.1):
        # Add Gaussian noise to the 'target' column
        targets = data['target'].values
        
        # Generate Gaussian noise
        noise = np.random.normal(loc=0.0, scale=noise_std, size=len(targets))
        
        # Add noise to the targets
        noisy_targets = targets + noise
        
        # You can optionally clip the values to ensure they stay within a reasonable range
        # For example, clip between the min and max of the original target values
        # noisy_targets = np.clip(noisy_targets, np.min(targets), np.max(targets))
        
        data['target'] = noisy_targets
        return data

    def save_to_csv(self, data: pd.DataFrame, filename):
        data.to_csv(filename, index=False)

def process_all_files_in_folder(input_folder, output_folder, noise_std=0.1):
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
            data_with_noise = data_processor.apply_gaussian_noise(data, noise_std)
            data_processor.save_to_csv(data_with_noise, output_filepath)
            print(f"Processed and saved: {filename}")

# Example usage
input_folder = "./data/standardized"  # Folder containing the CSV files
output_folder = "./data/noisy"    # Folder to save the processed files
noise_std = 3  # Standard deviation of the Gaussian noise

process_all_files_in_folder(input_folder, output_folder, noise_std)
