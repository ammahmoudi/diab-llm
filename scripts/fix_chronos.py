import os
import numpy as np
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s | %(message)s")

def calculate_rmse(prediction, ground_truth):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((prediction - ground_truth) ** 2))

def calculate_mae(prediction, ground_truth):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(prediction - ground_truth))

def calculate_mape(prediction, ground_truth):
    """Calculate Mean Absolute Percentage Error."""
    return np.mean(np.abs((prediction - ground_truth) / ground_truth)) * 100

def replace_bad_predictions(df, pred_columns, file_name, threshold_factor=3):
    """
    Detects and replaces extremely bad predictions using a threshold factor.

    Parameters:
        df (pd.DataFrame): The dataframe containing prediction values.
        pred_columns (list): List of column names with predicted values.
        file_name (str): Name of the CSV file being processed.
        threshold_factor (float): Multiplier for defining a bad prediction threshold.

    Returns:
        pd.DataFrame: The dataframe with corrected predictions.
    """
    for col in pred_columns:
        median_val = np.median(df[col].values)
        threshold = median_val * threshold_factor  # Define the extreme value threshold

        for i in range(1, len(df[col]) - 1):  # Avoid first and last row
            if df.at[i, col] > threshold:  # If the value is way too high
                new_value = (df.at[i - 1, col] + df.at[i + 1, col]) / 2
                logging.info(f"Fixed bad prediction in {file_name} | Column: {col} | Row: {i} | "
                             f"Original: {df.at[i, col]:.4f} -> New: {new_value:.4f} (Threshold: {threshold:.4f})")
                df.at[i, col] = new_value  # Replace with average of neighbors

    return df

def update_log_file(log_file, original_metrics, corrected_metrics):
    """
    Updates the log file with metric results for both original and corrected predictions.

    Parameters:
        log_file (str): Path to the log file.
        original_metrics (dict): Metrics for original predictions.
        corrected_metrics (dict): Metrics for corrected predictions.
    """
    if not os.path.exists(log_file):
        logging.warning(f"Log file {log_file} not found. Creating a new one.")
        open(log_file, 'w').close()

    with open(log_file, "a") as f:
        log_entry = (
            f"{logging.Formatter().formatTime(logging.makeLogRecord({}))} | INFO | "
            f"Original Metrics: {original_metrics}, Corrected Metrics: {corrected_metrics}\n"
        )
        f.write(log_entry)

    logging.info(f"Updated log file: {log_file} with new metrics.")

def process_csv_files(base_dir):
    """
    Processes CSV files ending with '_reformatted.csv' and not starting with 'smoothed_'.
    Fixes bad predictions, saves corrected data, and updates the log file.

    Parameters:
        base_dir (str): Base directory containing CSV files.
    """
    logging.info("Starting CSV processing...")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith("_reformatted.csv") and not file.startswith("smoothed_"):
                csv_path = os.path.join(root, file)
                log_file = os.path.join(root, "log.log")
                final_results_path = os.path.join(root, "final_results.csv")

                # Load CSV
                df = pd.read_csv(csv_path)

                # Identify true and predicted columns
                true_columns = [col for col in df.columns if "_true" in col]
                pred_columns = [col for col in df.columns if "_pred" in col and not col.startswith("smoothed_")]

                if not true_columns or not pred_columns:
                    logging.warning(f"Skipping {csv_path}: Missing required columns.")
                    continue

                # Compute original metrics
                original_rmse = calculate_rmse(df[pred_columns].values, df[true_columns].values)
                original_mae = calculate_mae(df[pred_columns].values, df[true_columns].values)
                original_mape = calculate_mape(df[pred_columns].values, df[true_columns].values)

                original_metrics = {
                    "rmse": round(original_rmse, 5),
                    "mae": round(original_mae, 5),
                    "mape": round(original_mape, 5),
                }

                # Correct bad predictions
                df_corrected = replace_bad_predictions(df.copy(), pred_columns, file)

                # Compute corrected metrics
                corrected_rmse = calculate_rmse(df_corrected[pred_columns].values, df_corrected[true_columns].values)
                corrected_mae = calculate_mae(df_corrected[pred_columns].values, df_corrected[true_columns].values)
                corrected_mape = calculate_mape(df_corrected[pred_columns].values, df_corrected[true_columns].values)

                corrected_metrics = {
                    "rmse": round(corrected_rmse, 5),
                    "mae": round(corrected_mae, 5),
                    "mape": round(corrected_mape, 5),
                }

                # Save corrected data
                df_corrected.to_csv(final_results_path, index=False)
                logging.info(f"Saved corrected results to {final_results_path}")

                # Update log file
                update_log_file(log_file, original_metrics, corrected_metrics)

    logging.info("CSV processing completed.")

# Example usage:
process_csv_files("./experiment_configs_chronos_training_inference/")
