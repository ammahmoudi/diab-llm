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


def setup_logging(log_dir="logs", program_name="log", log_level=absl_logging.INFO):
    """
    Sets up logging with Abseil for file output and Python logging for console with colors.

    Args:
        log_dir (str): Directory where log files are saved.
        program_name (str): Prefix for the log file names.
        log_level (int): Logging level (e.g., absl_logging.INFO, absl_logging.DEBUG).

    Returns:
        str: The directory where logs are saved.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Abseil logging for file
    absl_handler = absl_logging.get_absl_handler()
    absl_handler.use_absl_log_file(program_name=program_name, log_dir=log_dir)
    absl_logging.set_verbosity(log_level)

    # Add colored console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Configure root logger for console logging
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # Initialize log messages
    absl_logging.info(f"Logging initialized. Logs will be saved to: {log_dir}")
    absl_logging.info(f"Log level set to: {COLORS.get('INFO', '')}INFO{COLORS['RESET']}")

    return log_dir
