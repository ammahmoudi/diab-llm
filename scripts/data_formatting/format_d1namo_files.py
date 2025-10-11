import os
import pandas as pd


def standardize_d1namo_files(base_folder, save_path):
    """
    Standardize D1NAMO dataset files from structure:
    data/D1NAMO/001/train_data.csv, data/D1NAMO/001/test_data.csv

    Converting to standardized format with columns: item_id, timestamp, target
    """
    # Create standardized output folder
    standardized_folder = save_path
    os.makedirs(standardized_folder, exist_ok=True)

    # Loop through patient folders
    for patient_folder in os.listdir(base_folder):
        patient_path = os.path.join(base_folder, patient_folder)

        # Check if it's a directory and follows the patient ID pattern
        if os.path.isdir(patient_path):
            try:
                # Extract patient ID
                patient_id = patient_folder  # e.g., "001"

                # Process train and test files if they exist
                for file_type in ["train_data.csv", "test_data.csv"]:
                    file_path = os.path.join(patient_path, file_type)

                    if os.path.exists(file_path):
                        # Read CSV file
                        df = pd.read_csv(file_path)

                        # Rename columns
                        df.rename(
                            columns={"_ts": "timestamp", "_value": "target"},
                            inplace=True,
                        )

                        # Add patient ID column
                        df["item_id"] = patient_id

                        # Convert timestamps to standard format
                        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
                        df["timestamp"] = df["timestamp"].dt.strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )

                        # Reorder columns
                        df = df[["item_id", "timestamp", "target"]]

                        # Create standardized filename: e.g., 001_train.csv or 001_test.csv
                        data_type = "training" if "train" in file_type else "testing"
                        standardized_filename = f"{patient_id}-ws-{data_type}.csv"
                        standardized_file = os.path.join(
                            standardized_folder, standardized_filename
                        )

                        # Save to standardized folder
                        df.to_csv(standardized_file, index=False)

                        print(f"Processed: {patient_id} - {data_type}")

            except Exception as e:
                print(f"Error processing patient {patient_folder}: {e}")


# Example usage:
base_folder = "./data/D1NAMO"
save_path = "./data/d1namo_standardized"
standardize_d1namo_files(base_folder, save_path)
