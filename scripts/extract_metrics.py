import os
import re
import csv


def extract_metrics_to_csv(base_dir, output_csv):
    # Regex pattern to find the metric results line
    metrics_pattern = re.compile(r"Metric results: (\{.*\})")

    # List to store extracted results
    results = []
    csv_header = []  # Start with an empty header

    # Walk through all experiment folders
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "log.log":  # Targeting the log files
                log_path = os.path.join(root, file)

                # Extract experiment details from the folder structure
                path_parts = log_path.split(os.sep)  # Split by folder separators

                try:
                    # Extract the experiment folder and patient folder
                    experiment_folder = None
                    patient_folder = None

                    # Identify the experiment folder and patient folder
                    for part in path_parts:
                        if part.startswith("seed_"):
                            experiment_folder = part
                        elif part.startswith("patient_"):
                            patient_folder = part

                    if not experiment_folder or not patient_folder:
                        raise ValueError(
                            "Experiment or patient folder not found in path"
                        )

                    # Parse experiment folder details dynamically
                    experiment_details = experiment_folder.split("_")

                    # Extract the basic experiment parameters from folder structure
                    experiment_params = {}
                    for i in range(0, len(experiment_details), 2):
                        if i + 1 < len(experiment_details):
                            key = experiment_details[i]
                            value = experiment_details[i + 1]
                            experiment_params[key] = value

                    # Add patient_id from the patient folder
                    patient_id = patient_folder.split("_")[1]
                    experiment_params["patient_id"] = patient_id

                    # Read the log file and extract the last metrics line
                    with open(log_path, "r") as f:
                        lines = f.readlines()

                    # Search for the metrics line at the end of the file
                    metrics_line = None
                    for line in reversed(lines):
                        match = metrics_pattern.search(line)
                        if match:
                            metrics_line = match.group(1)  # Extract the dictionary part
                            break

                    if metrics_line:
                        # Convert the extracted string dictionary to a real dictionary
                        metrics = eval(metrics_line)  # Safe since we control input

                        # Dynamically generate the header if this is the first file
                        if not csv_header:
                            # Build header dynamically from the experiment_params keys
                            csv_header = list(experiment_params.keys()) + [
                                "rmse",
                                "mae",
                                "mape",
                            ]

                        # Store the extracted values in the results list
                        results.append(
                            list(experiment_params.values())
                            + [
                                metrics.get("rmse"),
                                metrics.get("mae"),
                                metrics.get("mape"),
                            ]
                        )

                        print(
                            f"Extracted metrics for {experiment_params['patient_id']} from {log_path}"
                        )
                    else:
                        # If no metrics found, append empty values to the results
                        if not csv_header:
                            # Build header dynamically from the experiment_params keys
                            csv_header = list(experiment_params.keys()) + [
                                "rmse",
                                "mae",
                                "mape",
                            ]

                        # Append the experiment data with empty metric values
                        results.append(list(experiment_params.values()) + ["", "", ""])
                        print(
                            f"No metrics found for {experiment_params['patient_id']} in {log_path}"
                        )

                except Exception as e:
                    print(f"Error processing {log_path}: {e}")

    # Save results to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)  # Write header
        writer.writerows(results)  # Write data

    print(f"Results saved to {output_csv}")


# Example usage:
# extract_metrics_to_csv("./experiment_configs/", "experiment_results.csv")
