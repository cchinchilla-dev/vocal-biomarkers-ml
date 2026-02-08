"""Unit tests for logging utilities module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


class TestSetupLogging:
    """Test suite for setup_logging function."""

    def test_setup_logging_returns_logger(self) -> None:
        """Test that setup_logging returns a logger."""
        from voice_analysis.utils.logging import setup_logging

        logger = setup_logging(level="WARNING", console=False)

        assert isinstance(logger, logging.Logger)

    def test_setup_logging_sets_level(self) -> None:
        """Test that logging level is set correctly."""
        from voice_analysis.utils.logging import setup_logging

        logger = setup_logging(level="DEBUG", console=False)

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path: Path) -> None:
        """Test logging to file."""
        from voice_analysis.utils.logging import setup_logging

        log_file = tmp_path / "test.log"
        logger = setup_logging(level="INFO", log_file=str(log_file), console=False)

        logger.info("Test message")

        assert log_file.exists()

    def test_setup_logging_custom_format(self) -> None:
        """Test custom log format."""
        from voice_analysis.utils.logging import setup_logging

        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(level="INFO", log_format=custom_format, console=False)

        assert logger is not None

    def test_setup_logging_suppresses_verbose_loggers(self) -> None:
        """Test that verbose loggers are suppressed."""
        from voice_analysis.utils.logging import setup_logging

        setup_logging(level="DEBUG", console=False)

        assert logging.getLogger("matplotlib").level == logging.WARNING
        assert logging.getLogger("PIL").level == logging.WARNING


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a logger."""
        from voice_analysis.utils.logging import get_logger

        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_same_name_returns_same_logger(self) -> None:
        """Test that same name returns same logger instance."""
        from voice_analysis.utils.logging import get_logger

        logger1 = get_logger("test_module")
        logger2 = get_logger("test_module")

        assert logger1 is logger2


class TestLoggingContext:
    """Test suite for LoggingContext class."""

    def test_logging_context_changes_level(self) -> None:
        """Test that LoggingContext temporarily changes level."""
        from voice_analysis.utils.logging import LoggingContext, setup_logging

        setup_logging(level="WARNING", console=False)
        root_logger = logging.getLogger()
        original_level = root_logger.level

        with LoggingContext("DEBUG"):
            assert root_logger.level == logging.DEBUG

        assert root_logger.level == original_level

    def test_logging_context_specific_logger(self) -> None:
        """Test LoggingContext with specific logger."""
        from voice_analysis.utils.logging import LoggingContext

        test_logger = logging.getLogger("test_specific")
        test_logger.setLevel(logging.WARNING)

        with LoggingContext("DEBUG", logger_name="test_specific"):
            assert test_logger.level == logging.DEBUG

        assert test_logger.level == logging.WARNING
