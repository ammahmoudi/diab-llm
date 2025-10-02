import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Union
from gluonts.dataset.arrow import ArrowWriter


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    timestamps: List[np.ndarray],
    compression: str = "lz4",
):
    """
    Store a given set of series into Arrow format at the specified path.
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and
        time_series.ndim == 2
    )

    # Prepare dataset with the actual timestamp
    dataset = [
        {"start": timestamps[i][0], "target": ts} 
        for i, ts in enumerate(time_series)
    ]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )


def load_csv_and_prepare_data(csv_path: str):
    """
    Load a CSV file, parse it, and prepare the data in the format
    suitable for GluonTS ArrowWriter.
    """
    # Read CSV data into a pandas DataFrame
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    # Ensure data is sorted by timestamp (if it's not already)
    df = df.sort_values("timestamp")

    # Group by 'item_id' to create separate time series for each item
    time_series = []
    timestamps = []
    for item_id, group in df.groupby("item_id"):
        # Extract the 'target' values and the 'timestamp' values
        target_values = group["target"].values
        timestamp_values = group["timestamp"].values

        # Store the time series and corresponding timestamps
        time_series.append(target_values)
        timestamps.append(timestamp_values)

    return time_series, timestamps


def convert_all_csv_in_directory(csv_directory: str, output_directory: str):
    """
    Convert all CSV files in a directory to Arrow format and save them in the specified output directory.
    The Arrow files will have the same name as the input CSV files but with a .arrow extension.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all files in the directory
    for filename in os.listdir(csv_directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(csv_directory, filename)

            # Load the data from the CSV file and prepare it
            time_series, timestamps = load_csv_and_prepare_data(csv_path)

            # Create the output Arrow file path with the same name but .arrow extension
            arrow_filename = f"{Path(filename).stem}.arrow"
            arrow_path = os.path.join(output_directory, arrow_filename)

            # Convert to GluonTS Arrow format and store in the output directory
            convert_to_arrow(arrow_path, time_series=time_series, timestamps=timestamps)
            print(f"Converted {filename} to {arrow_filename}")


if __name__ == "__main__":
    # Specify the directory where the CSV files are located
    csv_directory = "./data/d1namo_standardized"

    # Specify the directory where the Arrow files should be saved
    output_directory = "./data/d1namo_standardized"

    # Convert all CSV files in the folder to Arrow format
    convert_all_csv_in_directory(csv_directory, output_directory)
