import numpy as np
from absl import logging

def calculate_rmse(
        prediction,
        ground_truth
):
    squared_error = (prediction - ground_truth) ** 2
    mean_squared_error = np.mean(squared_error)
    rmse = np.sqrt(mean_squared_error)

    return rmse


def calculate_mae(
        prediction,
        ground_truth
):
    absolute_error = np.abs(prediction - ground_truth)
    mae = np.mean(absolute_error)

    return mae

def calculate_mape(
        prediction,
        ground_truth
):
    absolute_percentage_error = np.abs((prediction - ground_truth) / ground_truth)
    mape = np.mean(absolute_percentage_error)

    return mape