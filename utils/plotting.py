

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os


def plot_predictions_plotly(data, steps=None, save_path=None):
    """
    Plot the ground truth values and specified prediction steps for each timestamp, using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.

    Returns:
        None: Displays the plot.
    """
    # Extract ground truth and all prediction steps
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    # Iterate through rows and extract true and all pred values for each row
    for _, row in data.iterrows():
        true_values = [row[col] for col in data.columns if col.endswith("_true") and not pd.isna(row[col])]
        pred_values = [row[col] for col in data.columns if col.endswith("_pred") and not pd.isna(row[col])]

        if true_values:
            ground_truth.append(true_values[0])  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on the steps parameter
    if steps is not None:
        selected_steps = [i - 1 for i in steps if 1 <= i <= len(all_predictions)]  # Convert to 0-based indices
    else:
        selected_steps = range(len(all_predictions))

    # Create a range for the x-axis
    x_values = list(range(len(ground_truth)))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(go.Scatter(
        x=x_values,
        y=ground_truth,
        mode='lines+markers',
        name='Ground Truth',
        line=dict(dash='dash', color='green'),
        marker=dict(symbol='triangle-up', size=8, color='green')
    ))

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=predictions,
            mode='lines+markers',
            name=f'Predictions (Step {i+1})',
            line=dict(shape='linear')
        ))

    # Update layout
    fig.update_layout(
        title="Ground Truth vs Predictions (Selected Steps)",
        xaxis_title="Index",
        yaxis_title="Values",
        legend_title="Legend",
        template="plotly_white",
        height=600
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"ground_truth_vs_predictions_steps_{steps_str}"

        fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
    # Show the plot
         fig.show()


def plot_avg_predictions_plotly(data, save_path=None):
    """
    Plot the ground truth values and the averaged predictions for each timestamp, using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML.

    Returns:
        None: Displays the plot.
    """
    # Extract ground truth and predictions
    ground_truth = []
    avg_predictions = []  # To store the averaged predictions

    # Iterate through rows and extract true and pred values
    for _, row in data.iterrows():
        true_values = [row[col] for col in data.columns if col.endswith("_true") and not pd.isna(row[col])]
        pred_values = [row[col] for col in data.columns if col.endswith("_pred") and not pd.isna(row[col])]

        if true_values:
            ground_truth.append(true_values[0])  # Take the first available ground truth value

        if pred_values:
            avg_predictions.append(sum(pred_values) / len(pred_values))  # Average predictions

    # Create a range for the x-axis
    x_values = list(range(len(ground_truth)))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add ground truth trace
    fig.add_trace(go.Scatter(
        x=x_values,
        y=ground_truth,
        mode='lines+markers',
        name='Ground Truth',
        line=dict(dash='dash', color='green'),
        marker=dict(symbol='triangle-up', size=8, color='green')
    ))

    # Add averaged prediction trace
    fig.add_trace(go.Scatter(
        x=x_values,
        y=avg_predictions,
        mode='lines+markers',
        name='Averaged Predictions',
        line=dict(shape='linear', color='blue')
    ))

    # Update layout
    fig.update_layout(
        title="Ground Truth vs Averaged Predictions",
        xaxis_title="Index",
        yaxis_title="Values",
        legend_title="Legend",
        template="plotly_white",
        height=600
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        filename_base = "ground_truth_vs_avg_predictions"

        fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))
    else:
    # Show the plot
         fig.show()

def plot_predictions_matplotlib(data):
    """
    Plot the ground truth values and all predictions for each timestamp, accounting for overlapping windows.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_timestamp: Timestamp for time step i.
            - t_i_input: Input value for time step i (if available).
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.

    Returns:
        None: Displays the plot.
    """
    # Extract ground truth and all prediction steps
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    # Iterate through rows and extract true and all pred values for each row
    for _, row in data.iterrows():
        true_values = [row[col] for col in data.columns if col.endswith("_true") and not pd.isna(row[col])]
        pred_values = [row[col] for col in data.columns if col.endswith("_pred") and not pd.isna(row[col])]

        if true_values:
            ground_truth.append(true_values[0])  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Create a range for the x-axis
    x_values = range(len(ground_truth))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, ground_truth, label="Ground Truth", linestyle="--", marker="o", color="green")

    # Plot each prediction step
    for i, predictions in enumerate(all_predictions):
        plt.plot(x_values, predictions, label=f"Predictions (Step {i+1})", linestyle="-", marker="x")

    # Formatting
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Ground Truth vs Predictions (All Steps)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Show the plot
    plt.show()

