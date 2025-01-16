import os
import logging
from datetime import datetime
from absl import logging as absl_logging

# ANSI escape sequences for colored logs
COLORS = {
    "DEBUG": "\033[94m",    # Blue
    "INFO": "\033[92m",     # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "FATAL": "\033[95m",    # Magenta
    "RESET": "\033[0m"      # Reset
}

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for adding colors to console log outputs.
    """
    def format(self, record):
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)

class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured and formatted file logs.
    """
    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)

def setup_logging(base_log_dir="./logs", program_name="log", log_level=absl_logging.INFO):
    """
    Sets up logging with Abseil for file output and Python logging for console with colors.

    Args:
        base_log_dir (str): Base directory where log subdirectories are created.
        program_name (str): Prefix for the log file names.
        log_level (int): Logging level (e.g., absl_logging.INFO, absl_logging.DEBUG).

    Returns:
        str: The directory where logs for this run are saved.
    """
    # Create a timestamped subdirectory for the current run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(base_log_dir, f"logs_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)

    # Configure Abseil for file logging
    absl_handler = absl_logging.get_absl_handler()
    absl_handler.use_absl_log_file(program_name=program_name, log_dir=run_log_dir)
    absl_logging.set_verbosity(log_level)

    # Add Python console handler with colored formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Add structured formatting to Abseil's file handlers
    for handler in absl_logging.get_absl_logger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(StructuredFormatter())

    # Add console handler to Python's root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers to avoid duplication
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # Log initialization details
    absl_logging.info(f"Logging initialized. Logs for this run will be saved in: {run_log_dir}")
    absl_logging.info("Log level set to: INFO")

    return run_log_dir
