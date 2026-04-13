import logging
import logging.handlers
import sys
import os
from collections import deque
from typing import Optional

class LoggerHandler(logging.Handler):
    """A custom logging handler that keeps the most recent log entries in memory."""

    MAX_ENTRIES = 5000

    def __init__(self):
        super().__init__()
        self._entries: deque = deque(maxlen=self.MAX_ENTRIES)

    @property
    def log(self) -> str:
        return "\n\n".join(self._entries)

    def reset(self):
        """Clears the current log messages."""
        self._entries.clear()

    def emit(self, record):
        """Formats the log record and appends to the bounded deque."""
        log_entry = self.format(record)
        self._entries.append(log_entry)

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

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=3
        )
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
