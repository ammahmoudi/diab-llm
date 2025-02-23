import os
import glob
from reformat_t1dm_bg_data import reformat_t1dm_bg_data

def generate_features_labels(input_window_size, prediction_window_size):
    input_features = [f"BG_{{t-{i}}}" for i in range(input_window_size-1, 0, -1)] + ["BG_{t}"]
    labels = [f"BG_{{t+{i}}}" for i in range(1, prediction_window_size + 1)]
    return input_features, labels

def process_files(input_folder, output_folder, input_window_size, prediction_window_size):
    # Generate input features and labels
    input_features, labels = generate_features_labels(input_window_size, prediction_window_size)
    
    # Ensure the output directory exists
    formatted_folder = os.path.join(output_folder, f"{input_window_size}_{prediction_window_size}")
    os.makedirs(formatted_folder, exist_ok=True)
    
    # Find all CSV files in the input folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        save_path = os.path.join(formatted_folder, filename)
        
        print(f"Processing: {csv_path} -> {save_path}")
        print(input_features)
        print(labels)
        
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
    
    print(f"All files processed and saved in: {formatted_folder}")

if __name__ == '__main__':
    # User-defined parameters
    input_folder = '/home/amma/LLM-TIME/data/standardized/'
    output_folder = '/home/amma/LLM-TIME/data/formatted/'
    input_window_size = 6
    prediction_window_size = 9
    
    # Process all CSV files
    process_files(input_folder, output_folder, input_window_size, prediction_window_size)