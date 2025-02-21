import os
import pandas as pd

def standardize_files(folder_path,save_path):
    # Create a "standardized" folder inside the given folder
    standardized_folder = save_path
    os.makedirs(standardized_folder, exist_ok=True)

    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            try:
                # Extract patient ID from filename
                patient_id = filename.split("-")[0]
                
                # Read CSV file
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                
                # Rename columns
                df.rename(columns={'_ts': 'timestamp', '_value': 'target'}, inplace=True)
                
                # Add patient ID column
                df['item_id'] = patient_id
                
                # Convert timestamps to standard format
                df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
                df['timestamp'] = df['timestamp'].dt.strftime("%d-%m-%Y %H:%M:%S")

                # Reorder columns
                df = df[['item_id', 'timestamp', 'target']]
                
                # Save to standardized folder
                standardized_file = os.path.join(standardized_folder, filename)
                df.to_csv(standardized_file, index=False)
                
                print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage:
folder_path = './data/raw'
save_path ='./data/standardized'
standardize_files(folder_path,save_path)
