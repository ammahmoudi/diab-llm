import logging
import os

import pandas as pd

from data_processing.refromat_results import reformat_results
from utils.plotting import plot_avg_predictions_plotly, plot_predictions_plotly


def save_results_and_generate_plots(output_dir, predictions, targets, inputs, grouped_x_timestamps, grouped_y_timestamps):
    """
    Save prediction results, reformat them, and generate plots.

    :param output_dir: Directory to save the predictions, formatted results, and plots.
    :param predictions: Array of predictions.
    :param targets: Array of ground truth values.
    :param inputs: Array of input data.
    :param grouped_x_timestamps: List of grouped x timestamps as strings.
    :param grouped_y_timestamps: List of grouped y timestamps as strings.
    """
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV
    save_path = os.path.join(output_dir, "inference_results.csv")
    results = {
        "x_timestamps": grouped_x_timestamps,
        "y_timestamps": grouped_y_timestamps,
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
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)

    # Reformat and save the results
    reformatted_path = save_path.replace(".csv", "_reformatted.csv")
    reformated_results_df=reformat_results(save_path, output_csv_path=reformatted_path)
    logging.info(f"Results saved to {save_path} and reformatted results to {reformatted_path}")

    # Generate and save plots
    plots_dir = os.path.join(output_dir, "plots")
    plot_predictions_path = os.path.join(plots_dir, "predictions")
    plot_avg_predictions_path = os.path.join(plots_dir, "avg_predictions")

    plot_predictions_plotly(reformated_results_df, save_path=plot_predictions_path)
    plot_avg_predictions_plotly(reformated_results_df, save_path=plot_avg_predictions_path)

    logging.info(f"Plots saved to {plots_dir}")