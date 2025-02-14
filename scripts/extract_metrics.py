import os
import re
import csv

# Base directory containing all experiment logs
base_dir = "./experiment_configs/"
output_csv = "experiment_results.csv"

# Regex pattern to find the metric results line
metrics_pattern = re.compile(r"Metric results: (\{.*\})")

# Prepare the CSV header
csv_header = [
    "seed", "model", "dtype", "mode", "context_length", "prediction_length",
    "patient_id", "rmse", "mae", "mape"
]

# List to store extracted results
results = []

# Walk through all experiment folders
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "log.log":  # Targeting the log files
            log_path = os.path.join(root, file)

            # Extract experiment details from the folder structure
            path_parts = log_path.split(os.sep)  # Split by folder separators

            try:
                # Automatically find the main experiment folder
                experiment_folder = next(
                    part for part in path_parts if part.startswith("seed_")
                )

                # Extract values dynamically
                seed = experiment_folder.split("_")[1]
                model = experiment_folder.split("_")[3]
                dtype = experiment_folder.split("_")[5]
                mode = experiment_folder.split("_")[7]
                context_length = experiment_folder.split("_")[9]
                prediction_length = experiment_folder.split("_")[11]

                # Extract patient ID from folder structure
                patient_folder = next(
                    part for part in path_parts if part.startswith("patient_")
                )
                patient_id = patient_folder.split("_")[1]

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

                    # Store the extracted values in the results list
                    results.append([
                        seed, model, dtype, mode, context_length, prediction_length,
                        patient_id, metrics.get("rmse"), metrics.get("mae"), metrics.get("mape")
                    ])

                    print(f"Extracted metrics for {patient_id} from {log_path}")

            except Exception as e:
                print(f"Error processing {log_path}: {e}")

# Save results to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)  # Write header
    writer.writerows(results)  # Write data

print(f"Results saved to {output_csv}")
