import logging
import sys
import os
import transformers
from typing import Optional

def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

class LoggerHandler(logging.Handler):
    """A custom logging handler designed for specific logging needs."""

    def __init__(self):
        super().__init__()
        self.log = ""

    def reset(self):
        """Clears the current log messages."""
        self.log = ""

    def emit(self, record):
        """Formats the log record."""
        log_entry = self.format(record)
        self.log += f"{log_entry}\n\n"

def get_logger(name: str, level=logging.INFO, log_file=None) -> logging.Logger:
    """Creates and returns a logger with a predefined configuration.
    Args:
        name (str): The name of the logger.
        level: The threshold level for the logger. Defaults to logging.INFO.
        log_file (str, optional): Path to a log file. If specified, logs will also be written to this file.

    Returns:
        logging.Logger: A configured logger.
    """

    name = name.split(".")[0]  # Ensures that the logger name is not too long.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # Prevents log messages from propagating to the root logger.

    # Check if handlers already exist to avoid duplicate logging
    if logger.handlers:
        return logger

    # Defines the format for log messages.
    formatter = logging.Formatter(
        # fmt="%(asctime)s | %(levelname)s | %(name)s : [Line: %(lineno)d] - %(message)s", # TODO
        fmt="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Sets up console logging.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Sets up file logging if a log file path is provided.
    if log_file:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)  # This will create the directory if it does not exist

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
def reset_logging():
    """Resets the logging configuration by removing all handlers and filters from the root logger."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    for filter in root.filters[:]:
        root.removeFilter(filter)
