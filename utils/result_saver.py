import logging
import os
import pandas as pd

from data_processing.refromat_results import reformat_results
from utils.plotting import plot_avg_predictions_plotly, plot_predictions_plotly

def generate_run_title(data_settings, llm_settings):
    # Extract dataset name from the train data path
    train_data_path = data_settings.get('path_to_train_data', '')
    dataset_name = train_data_path.split('/')[-1].split('-')[0] if train_data_path else 'unknown'

    method = llm_settings.get('method', 'unknown')
    model = llm_settings.get('model', llm_settings.get('llm_model', 'unknown')).replace('/','-')
    mode = llm_settings.get('mode', 'unknown')

    return f"patient{dataset_name}_{method}_{model}_{mode}"

def save_results_and_generate_plots(
    output_dir, predictions, targets, inputs, grouped_x_timestamps=None, grouped_y_timestamps=None, name=None
):
    """
    Save prediction results, reformat them, and generate plots.

    :param output_dir: Directory to save the predictions, formatted results, and plots.
    :param predictions: Array of predictions.
    :param targets: Array of ground truth values.
    :param inputs: Array of input data.
    :param grouped_x_timestamps: (Optional) List of grouped x timestamps as strings.
    :param grouped_y_timestamps: (Optional) List of grouped y timestamps as strings.
    :param name: (Optional) A name to use in the plot and file names.
    """
    # Debug: log the shapes of the inputs
    logging.debug(f"Shape of predictions: {predictions.shape}")
    logging.debug(f"Shape of targets: {targets.shape}")
    logging.debug(f"Shape of inputs: {inputs.shape}")

    if grouped_x_timestamps is not None:
        logging.debug(f"Number of grouped_x_timestamps: {len(grouped_x_timestamps)}")
    else:
        logging.debug("grouped_x_timestamps not provided.")

    if grouped_y_timestamps is not None:
        logging.debug(f"Number of grouped_y_timestamps: {len(grouped_y_timestamps)}")
    else:
        logging.debug("grouped_y_timestamps not provided.")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build the results dictionary
    results = {
        "inputs": [
            ", ".join(map(str, inputs[i].flatten())) for i in range(len(inputs))
        ],
        "ground_truth": [
            ", ".join(map(str, targets[i].flatten())) for i in range(len(targets))
        ],
        "predictions": [
            ", ".join(map(str, predictions[i].flatten())) for i in range(len(predictions))
        ],
    }

    # Add timestamps if provided
    if grouped_x_timestamps is not None:
        results["x_timestamps"] = [
            ", ".join(map(str, grouped_x_timestamps[i])) for i in range(len(grouped_x_timestamps))
        ]
    if grouped_y_timestamps is not None:
        results["y_timestamps"] = [
            ", ".join(map(str, grouped_y_timestamps[i])) for i in range(len(grouped_y_timestamps))
        ]

    # Use the provided name or default to "inference"
    file_name_prefix = name if name else "inference"

    # Save results to CSV
    save_path = os.path.join(output_dir, f"{file_name_prefix}_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    logging.info(f"Results saved to {save_path}")

    # Reformat and save the results
    reformatted_path = save_path.replace(".csv", "_reformatted.csv")
    reformated_results_df = reformat_results(save_path, output_csv_path=reformatted_path)
    logging.info(f"Reformatted results saved to {reformatted_path}")

    # Generate and save plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_predictions_path = os.path.join(plots_dir, f"{file_name_prefix}_predictions")
    plot_avg_predictions_path = os.path.join(plots_dir, f"{file_name_prefix}_avg_predictions")

    plot_predictions_plotly(reformated_results_df, save_path=plot_predictions_path,name=name)
    plot_avg_predictions_plotly(reformated_results_df, save_path=plot_avg_predictions_path,name=name)

    logging.info(f"Plots saved to {plots_dir}")




import pandas as pd

def convert_window_to_single_prediction(input_file, output_file="single_step_predictions.csv"):
    """
    Converts windowed time-series predictions into a single-step format by:
    1. Extracting the first step's true and predicted values.
    2. Extracting the last row's true and predicted values, ensuring all steps are captured.
    
    Saves the result to a new CSV file with "true" and "pred" columns.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the processed CSV file (default: "single_step_predictions.csv").
    """
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Identify true and pred columns
    true_cols = [col for col in df.columns if "_true" in col]
    pred_cols = [col for col in df.columns if "_pred" in col]

    # Ensure there are true and pred columns
    if not true_cols or not pred_cols:
        print("No valid columns found in the CSV.")
        return

    # Select only the first true and predicted column (first step)
    selected_true = true_cols[0]
    selected_pred = pred_cols[0]

    # Extract the first step's true and predicted values
    first_step_df = df[[selected_true, selected_pred]].copy()
    first_step_df.columns = ["true", "pred"]  # Rename columns

    # Extract the last row
    last_row = df.iloc[-1]

    # Separate the last row into true and pred values
    last_true_values = last_row[true_cols].reset_index(drop=True)
    last_pred_values = last_row[pred_cols].reset_index(drop=True)

    # Create DataFrames for true and pred values
    last_true_df = pd.DataFrame(last_true_values, columns=["true"])
    last_pred_df = pd.DataFrame(last_pred_values, columns=["pred"])

    # Concatenate first step data with last row data
    result_df = pd.concat([first_step_df, last_true_df, last_pred_df], ignore_index=True)

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    print(f"Converted window predictions saved to {output_file}")

# Example Usage:
# convert_window_to_single_prediction("your_file.csv", "single_step_predictions.csv")
