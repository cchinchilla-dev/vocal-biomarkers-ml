"""
Logging utilities module.

This module provides utilities for configuring logging
throughout the voice analysis pipeline.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_format: str | None = None,
    log_file: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """
    Configure logging for the application.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_format : str, optional
        Log message format string.
    log_file : str, optional
        Path to log file. If None, no file logging.
    console : bool
        Whether to log to console.

    Returns
    -------
    Logger
        Configured root logger.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("librosa").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).

    Returns
    -------
    Logger
        Logger instance.
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporarily changing log level.

    Parameters
    ----------
    level : str
        Temporary log level.
    logger_name : str, optional
        Specific logger to modify. If None, modifies root.

    Examples
    --------
    >>> with LoggingContext("DEBUG"):
    ...     # Debug messages will be shown
    ...     pass
    """

    def __init__(self, level: str, logger_name: str | None = None):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        self.original_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)
        return False


class ProgressLogger:
    """
    Logger for tracking progress of long-running operations.

    Parameters
    ----------
    total : int
        Total number of items to process.
    description : str
        Description of the operation.
    log_interval : int
        Log progress every N items.

    Examples
    --------
    >>> progress = ProgressLogger(100, "Processing")
    >>> for i in range(100):
    ...     progress.update(i + 1)
    """

    def __init__(
        self,
        total: int,
        description: str = "Progress",
        log_interval: int = 10,
    ):
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.current = 0

    def update(self, current: int | None = None) -> None:
        """Update progress."""
        if current is not None:
            self.current = current
        else:
            self.current += 1

        if self.current % self.log_interval == 0 or self.current == self.total:
            percentage = (self.current / self.total) * 100
            self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")

    def complete(self) -> None:
        """Mark operation as complete."""
        self.logger.info(f"{self.description}: Complete ({self.total} items)")
