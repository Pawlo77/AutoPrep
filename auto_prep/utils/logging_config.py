import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

# ANSI color codes
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
    "RESET": "\033[0m",  # Reset color
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname and timing information."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a logging record into a string.

        Args:
            record (logging.LogRecord): The logging record to be formatted.

        Returns:
            str: The formatted string representation of the logging record.
        """
        # Add color to levelname
        levelname = record.levelname
        record.levelname = f"{COLORS[levelname]}{levelname:<8}{COLORS['RESET']}"

        # Add elapsed time if available
        if hasattr(record, "elapsed_time"):
            record.msg = f"{record.msg} ({record.elapsed_time:.2f}s)"

        return super().format(record)


class TimedLogger(logging.Logger):
    """Logger subclass that tracks operation timing."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        """
        Initialize the TimedLogger.

        Args:
            name (str): The name of the logger.
            level (int, optional): The logging level. Defaults to logging.NOTSET.
        """
        super().__init__(name, level)
        self._operation_start: Optional[float] = None
        self._current_operation: Optional[str] = None

    def start_operation(self, operation: str) -> None:
        """
        Initiates the timing of a specific operation.

        This method marks the beginning of an operation and starts the timer.
        It is used to track the duration of a specific operation or task.

        Args:
            operation (str): The name or description of the operation being timed.
        """
        self._operation_start = time.time()
        self._current_operation = operation
        self.debug(f"→ {operation}")

    def end_operation(self) -> None:
        """End timing the current operation and log the elapsed time."""
        if self._operation_start and self._current_operation:
            elapsed = time.time() - self._operation_start
            self.debug(f"✓ {self._current_operation}", extra={"elapsed_time": elapsed})
            self._operation_start = None
            self._current_operation = None


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO
LOG_DIR = None  # don't save logs for now


def setup_logger(name: str) -> TimedLogger:
    """
    Sets up a logger with both file and console handlers.

    Args:
        name (str): The name of the logger.

    Returns:
        TimedLogger: An instance of the TimedLogger class, which is a subclass
            of the standard Python logger that adds timing functionality.
    """
    logging.setLoggerClass(TimedLogger)

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        if LOG_DIR is not None:
            os.makedirs(LOG_DIR, exist_ok=True)
            file_handler = RotatingFileHandler(
                f"{LOG_DIR}/{name.split('.')[-1]}.log",
                maxBytes=1024 * 1024,  # 1MB
                backupCount=5,
            )
            file_handler.setLevel(LOG_LEVEL)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(console_handler)

    return logger
