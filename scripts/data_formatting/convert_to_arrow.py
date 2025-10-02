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

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
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

    The expected CSV format:
    item_id,timestamp,target
    540,04-07-2027 00:01:44,254
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


if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_path = "./data/standardized/570-ws-training.csv"

    # Load the data from the CSV file and prepare it
    time_series, timestamps = load_csv_and_prepare_data(csv_path)

    # Convert to GluonTS Arrow format and store in a file
    convert_to_arrow("./data/standardized/570-ws-training.arrow", time_series=time_series, timestamps=timestamps)
