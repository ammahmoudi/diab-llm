

import os
from matplotlib import pyplot as plt

plt.switch_backend('agg')


def plot_predictions_with_overlays(
    timestamps,
    inputs,
    ground_truth,
    predictions,
    context_length,
    save_path=None,
    title="Prediction Overlays"
):
    """
    Plot predictions with overlays of context, ground truth, and predictions.
    
    :param timestamps: Array of timestamps for x-axis.
    :param inputs: Input data (context) used for predictions.
    :param ground_truth: Ground truth (actual values).
    :param predictions: Predicted values.
    :param context_length: Length of the context used for predictions.
    :param save_path: File path to save the plot (optional).
    :param title: Title of the plot.
    """
    num_plots = min(5, len(predictions))  # Limit to 5 plots for readability

    plt.figure(figsize=(12, 6))
    for idx in range(num_plots):
        context_end = context_length

        # Flatten or reshape inputs to ensure compatibility with timestamps
        input_context = inputs[idx].flatten()
        if len(input_context) != context_end:
            input_context = input_context[:context_end]

        plt.plot(
            timestamps[:context_end],
            input_context,
            label=f"Input Context {idx + 1}",
            linestyle='-',
            color="gray",
            alpha=0.7,
        )

        # Plot ground truth
        plt.plot(
            timestamps[context_end:context_end + len(ground_truth[idx])],
            ground_truth[idx].flatten(),
            label=f"Ground Truth {idx + 1}",
            linestyle='--',
            color="blue",
            alpha=0.8,
        )

        # Plot predictions
        plt.plot(
            timestamps[context_end:context_end + len(predictions[idx])],
            predictions[idx].flatten(),
            label=f"Prediction {idx + 1}",
            linestyle='-',
            color="red",
        )

    plt.title(title)
    plt.xlabel("Timestamps")
    plt.ylabel("Values")
    plt.legend(loc="upper left", fontsize="small")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
