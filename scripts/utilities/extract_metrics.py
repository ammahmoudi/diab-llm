import os
import re
import csv
import logging
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import math


def sanitize_metric(value):
    try:
        # Handle string representations of numpy values like np.float32(13.443349)
        if isinstance(value, str) and "np.float" in value:
            # Extract the numeric value from inside parentheses
            match = re.search(r"\((.*?)\)", value)
            if match:
                value = match.group(1)

        val = float(value)
        if math.isinf(val) or math.isnan(val):
            return "NaN"
        return str(val)
    except (ValueError, TypeError):
        return "NaN"


def extract_metrics_to_csv(base_dir, output_csv):
    logging.info("Starting metrics extraction.")

    # Regex pattern to find the metric results line
    metrics_pattern = re.compile(r"Metric results: (\{.*\})")
    log_datetime_pattern = re.compile(r"logs_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")

    # List to store extracted results
    results = []
    csv_header = []  # Start with an empty header
    existing_rows = set()

    # Check if the file already exists and load existing rows (excluding the header)
    file_exists = os.path.exists(output_csv)
    if file_exists and os.stat(output_csv).st_size > 0:
        with open(output_csv, "r", newline="") as f:
            reader = csv.reader(f)
            csv_header = next(reader, None)  # Read the header
            for row in reader:
                existing_rows.add(tuple(row))  # Store existing rows
        logging.info(f"Loaded {len(existing_rows)} existing rows from {output_csv}.")

    # Walk through all experiment folders
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "log.log":  # Targeting the log files
                log_path = os.path.join(root, file)
                logging.info(f"Processing log file: {log_path}")

                # Extract log date-time from the log folder path
                log_datetime_match = log_datetime_pattern.search(log_path)
                log_datetime = (
                    log_datetime_match.group(1) if log_datetime_match else "Unknown"
                )

                # Extract experiment details from the folder structure
                path_parts = log_path.split(os.sep)

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

                    # Parse experiment folder details using proper key-value extraction
                    # Expected format: seed_VALUE_model_VALUE_dtype_VALUE_mode_VALUE[_inference]_context_VALUE_pred_VALUE
                    experiment_details = experiment_folder.split("_")
                    
                    experiment_params = {}
                    i = 0
                    while i < len(experiment_details) - 1:  # Ensure we don't go out of bounds
                        key = experiment_details[i]
                        
                        if key in ["seed", "model", "dtype", "mode", "context", "pred"]:
                            if key == "model":
                                # Handle multi-part model names like amazon-chronos-t5-base
                                # Collect parts until we hit the next known key
                                value_parts = []
                                j = i + 1
                                while j < len(experiment_details) and experiment_details[j] not in ["dtype", "mode", "context", "pred"]:
                                    value_parts.append(experiment_details[j])
                                    j += 1
                                value = "-".join(value_parts)  # Join with hyphens for model names
                                i = j - 1
                            elif key == "mode":
                                # Handle mode which might be followed by "inference"
                                value = experiment_details[i + 1]
                                # Check if next token is "inference" and append it
                                if i + 2 < len(experiment_details) and experiment_details[i + 2] == "inference":
                                    value += "_inference"
                                    i += 1  # Skip the "inference" token
                                i += 1
                            else:
                                # Simple key-value pair
                                value = experiment_details[i + 1]
                                i += 1
                            
                            experiment_params[key] = value
                        
                        i += 1

                    # Add patient_id from the patient folder
                    patient_id = patient_folder.split("_")[1]
                    experiment_params["patient_id"] = patient_id
                    experiment_params["log_datetime"] = (
                        log_datetime  # Add the extracted timestamp
                    )

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
                        import ast

                        sanitized_str = (
                            metrics_line.replace("inf", '"inf"')
                            .replace("-inf", '"-inf"')
                            .replace("nan", '"nan"')
                        )
                        sanitized_str = re.sub(
                            r"np\.float\d*\(([\d\.]+)\)", r'"\1"', sanitized_str
                        )
                        try:
                            metrics = ast.literal_eval(sanitized_str)
                        except Exception as e:
                            logging.error(f"Error parsing metrics in {log_path}: {e}")
                            continue

                        # Generate proper header with meaningful column names
                        if not csv_header:
                            # Create header with proper column names in logical order
                            csv_header = [
                                "seed", "model", "dtype", "mode", 
                                "context_length", "pred_length", 
                                "patient_id", "log_datetime",
                                "rmse", "mae", "mape"
                            ]

                        # Store the extracted values in the correct order to match header
                        result_row = (
                            str(experiment_params.get("seed", "")),
                            str(experiment_params.get("model", "")),
                            str(experiment_params.get("dtype", "")),
                            str(experiment_params.get("mode", "")),
                            str(experiment_params.get("context", "")),
                            str(experiment_params.get("pred", "")),
                            str(experiment_params.get("patient_id", "")),
                            str(experiment_params.get("log_datetime", "")),
                            sanitize_metric(metrics.get("rmse")),
                            sanitize_metric(metrics.get("mae")),
                            sanitize_metric(metrics.get("mape"))
                        )

                        if result_row not in existing_rows:  # Avoid duplicate rows
                            results.append(result_row)
                            existing_rows.add(result_row)
                            logging.info(f"New data added: {result_row}")
                        else:
                            logging.info("Duplicate entry found, skipping.")
                    else:
                        logging.warning(f"No metric results found in {log_path}")

                except Exception as e:
                    logging.error(f"Error processing {log_path}: {e}")

    # Save results to CSV (Append Mode)
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)

        if (
            not file_exists or os.stat(output_csv).st_size == 0
        ):  # Write header only if the file is new
            writer.writerow(csv_header)
            logging.info(f"CSV header written: {csv_header}")

        if results:  # Only write if there are new results
            writer.writerows(results)  # Append new rows only if they are not duplicates
            logging.info(f"Appended {len(results)} new rows to {output_csv}")
        else:
            logging.info("No new data to append.")

    logging.info(f"Results saved to {output_csv}")


# Example usage:
# extract_metrics_to_csv("./experiment_configs_time_llm_inference/", "experiment_results_test.csv")
