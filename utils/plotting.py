

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

def plot_clarke_error_grid_plotly(data, steps=None, save_path=None):
    """
    Plot a Clarke Error Grid using Plotly to assess the accuracy of predicted values against ground truth.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.

    Returns:
        None: Displays the Clarke Error Grid plot.
    """
    # Extract true and predicted values
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    for _, row in data.iterrows():
        true_values = [row[col] for col in data.columns if col.endswith("_true") and not pd.isna(row[col])]
        pred_values = [row[col] for col in data.columns if col.endswith("_pred") and not pd.isna(row[col])]

        if true_values:
            ground_truth.append(true_values[0])  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on steps
    if steps is not None:
        selected_steps = [i - 1 for i in steps if 1 <= i <= len(all_predictions)]
    else:
        selected_steps = range(len(all_predictions))

    # Initialize Plotly figure
    fig = go.Figure()

    # Add Clarke Error Grid regions
    max_val = max(max(ground_truth), max([max(pred) for pred in all_predictions]))
    x = np.linspace(0, max_val, 100)

    # Add region A (within 20% of true values)
    fig.add_trace(go.Scatter(
        x=x, y=1.2 * x, mode="lines", fill="tonexty", line=dict(color="green", dash="dash"),
        name="Region A Upper (20%)", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.8 * x, mode="lines", fill="none", line=dict(color="green", dash="dash"),
        name="Region A Lower (20%)"
    ))

    # Add region B (clinically acceptable outside region A)
    fig.add_trace(go.Scatter(
        x=x, y=1.3 * x, mode="lines", fill="tonexty", line=dict(color="yellow", dash="dash"),
        name="Region B Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.7 * x, mode="lines", fill="none", line=dict(color="yellow", dash="dash"),
        name="Region B Lower"
    ))

    # Add region C/D (over-treatment and under-treatment)
    fig.add_trace(go.Scatter(
        x=x, y=max_val * np.ones_like(x), mode="lines", fill="tonexty", line=dict(color="red", dash="dash"),
        name="Region C", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.7 * x, mode="lines", fill="tonexty", line=dict(color="red", dash="dash"),
        name="Region D", showlegend=False
    ))

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(go.Scatter(
            x=ground_truth,
            y=predictions,
            mode='markers',
            name=f'Predictions (Step {i+1})',
            marker=dict(size=8, symbol='circle')
        ))

    # Update layout
    fig.update_layout(
        title="Clarke Error Grid",
        xaxis_title="True Values",
        yaxis_title="Predicted Values",
        template="plotly_white",
        height=800,
        width=800
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"clarke_error_grid_steps_{steps_str}"

        fig.write_image(os.path.join(save_path, f"{filename_base}.png"))
        fig.write_image(os.path.join(save_path, f"{filename_base}.svg"))
        fig.write_html(os.path.join(save_path, f"{filename_base}.html"))

    else:
        # Show the plot
        fig.show()

def plot_surveillance_error_grid(data, steps=None, save_path=None):
    """
    Plot a Surveillance Error Grid (SEG) for blood glucose predictions using Plotly.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns structured as follows:
            - t_i_true: Ground truth value for time step i.
            - t_i_pred: Prediction value for time step i.
        steps (list, optional): List of prediction step indices to include (1-based). If None, include all steps.
        save_path (str, optional): Directory to save the plot. Saves as PNG, SVG, and HTML with filenames indicating included steps.

    Returns:
        None: Displays the SEG plot.
    """
    # Extract true and predicted values
    ground_truth = []
    all_predictions = []  # List of lists to store predictions for each step

    for _, row in data.iterrows():
        true_values = [row[col] for col in data.columns if col.endswith("_true") and not pd.isna(row[col])]
        pred_values = [row[col] for col in data.columns if col.endswith("_pred") and not pd.isna(row[col])]

        if true_values:
            ground_truth.append(true_values[0])  # Take the first available ground truth value

        for i, pred in enumerate(pred_values):
            if len(all_predictions) <= i:
                all_predictions.append([])  # Add a new list for this prediction step
            all_predictions[i].append(pred)

    # Filter predictions based on steps
    if steps is not None:
        selected_steps = [i - 1 for i in steps if 1 <= i <= len(all_predictions)]
    else:
        selected_steps = range(len(all_predictions))

    # Initialize Plotly figure
    fig = go.Figure()

    # Define Surveillance Error Grid regions (specific to blood glucose levels)
    max_val = max(max(ground_truth), max([max(pred) for pred in all_predictions]))
    x = np.linspace(0, max_val, 100)

    # Region A: Clinically insignificant
    fig.add_trace(go.Scatter(
        x=x, y=1.2 * x, mode="lines", fill="tonexty", line=dict(color="green", dash="dash"),
        name="Region A Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.8 * x, mode="lines", fill="none", line=dict(color="green", dash="dash"),
        name="Region A Lower"
    ))

    # Region B: Minimal risk
    fig.add_trace(go.Scatter(
        x=x, y=1.5 * x, mode="lines", fill="tonexty", line=dict(color="yellow", dash="dash"),
        name="Region B Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.6 * x, mode="lines", fill="none", line=dict(color="yellow", dash="dash"),
        name="Region B Lower"
    ))

    # Region C: Moderate risk
    fig.add_trace(go.Scatter(
        x=x, y=2 * x, mode="lines", fill="tonexty", line=dict(color="orange", dash="dash"),
        name="Region C Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.5 * x, mode="lines", fill="none", line=dict(color="orange", dash="dash"),
        name="Region C Lower"
    ))

    # Region D: High risk
    fig.add_trace(go.Scatter(
        x=x, y=max_val * np.ones_like(x), mode="lines", fill="tonexty", line=dict(color="red", dash="dash"),
        name="Region D Upper", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x, y=0.4 * x, mode="lines", fill="tonexty", line=dict(color="red", dash="dash"),
        name="Region D Lower", showlegend=False
    ))

    # Add prediction traces for each selected step
    for i in selected_steps:
        predictions = all_predictions[i]
        fig.add_trace(go.Scatter(
            x=ground_truth,
            y=predictions,
            mode='markers',
            name=f'Predictions (Step {i+1})',
            marker=dict(size=8, symbol='circle')
        ))

    # Update layout
    fig.update_layout(
        title="Surveillance Error Grid for Blood Glucose",
        xaxis_title="True Glucose Values (mg/dL)",
        yaxis_title="Predicted Glucose Values (mg/dL)",
        template="plotly_white",
        height=800,
        width=800
    )

    # Save the plot if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
        steps_str = "all" if steps is None else "_".join(map(str, steps))
        filename_base = f"surveillance_error_grid_steps_{steps_str}"

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

