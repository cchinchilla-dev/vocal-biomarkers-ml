"""
Data validation module.

This module provides validators for ensuring data integrity
and quality throughout the analysis pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validator for voice analysis datasets.

    This class provides methods for validating data integrity,
    checking for expected columns, and ensuring data quality.
    """

    REQUIRED_COLUMNS = {"Record", "Diagnosed"}
    VALID_DIAGNOSES = {"Control", "Pathological"}
    VALID_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

    def validate_dataset(self, dataset: pd.DataFrame) -> ValidationResult:
        """
        Perform comprehensive validation of a dataset.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to validate.

        Returns
        -------
        ValidationResult
            Result object containing validation status and messages.
        """
        result = ValidationResult()

        # Check required columns
        self._check_required_columns(dataset, result)

        # Check for missing values
        self._check_missing_values(dataset, result)

        # Check diagnosis values
        self._check_diagnosis_values(dataset, result)

        # Check for duplicate records
        self._check_duplicates(dataset, result)

        # Check numeric columns
        self._check_numeric_columns(dataset, result)

        return result

    def _check_required_columns(
        self, dataset: pd.DataFrame, result: "ValidationResult"
    ) -> None:
        """Check that all required columns are present."""
        missing = self.REQUIRED_COLUMNS - set(dataset.columns)
        if missing:
            result.add_error(f"Missing required columns: {missing}")
        else:
            result.add_success("All required columns present")

    def _check_missing_values(
        self, dataset: pd.DataFrame, result: "ValidationResult"
    ) -> None:
        """Check for missing values in the dataset."""
        missing_counts = dataset.isnull().sum()
        columns_with_missing = missing_counts[missing_counts > 0]

        if not columns_with_missing.empty:
            result.add_warning(
                f"Columns with missing values: {dict(columns_with_missing)}"
            )
        else:
            result.add_success("No missing values found")

    def _check_diagnosis_values(
        self, dataset: pd.DataFrame, result: "ValidationResult"
    ) -> None:
        """Check that diagnosis values are valid."""
        if "Diagnosed" not in dataset.columns:
            return

        unique_diagnoses = set(dataset["Diagnosed"].unique())
        invalid = unique_diagnoses - self.VALID_DIAGNOSES

        if invalid:
            result.add_error(f"Invalid diagnosis values: {invalid}")
        else:
            result.add_success("Diagnosis values are valid")

    def _check_duplicates(
        self, dataset: pd.DataFrame, result: "ValidationResult"
    ) -> None:
        """Check for duplicate records."""
        if "Record" not in dataset.columns:
            return

        duplicates = dataset["Record"].duplicated().sum()
        if duplicates > 0:
            result.add_warning(f"Found {duplicates} duplicate records")
        else:
            result.add_success("No duplicate records found")

    def _check_numeric_columns(
        self, dataset: pd.DataFrame, result: "ValidationResult"
    ) -> None:
        """Check numeric columns for invalid values."""
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if np.isinf(dataset[col]).any():
                result.add_warning(f"Column '{col}' contains infinite values")

            if (dataset[col] < 0).any() and col.startswith(("Pr", "MFCC")):
                # Some features shouldn't be negative
                pass  # This depends on the specific feature

        result.add_success(f"Checked {len(numeric_cols)} numeric columns")


class AudioFileValidator:
    """
    Validator for audio files.

    This class provides methods for validating audio file
    existence, format, and basic properties.
    """

    VALID_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
    MIN_DURATION_SECONDS = 1.0
    MAX_DURATION_SECONDS = 10.0

    def validate_file(self, filepath: Path) -> ValidationResult:
        """
        Validate a single audio file.

        Parameters
        ----------
        filepath : Path
            Path to the audio file.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        result = ValidationResult()

        # Check file exists
        if not filepath.exists():
            result.add_error(f"File not found: {filepath}")
            return result

        # Check extension
        if filepath.suffix.lower() not in self.VALID_EXTENSIONS:
            result.add_error(f"Invalid audio format: {filepath.suffix}")
            return result

        # Check file size
        file_size = filepath.stat().st_size
        if file_size == 0:
            result.add_error(f"Empty file: {filepath}")
            return result

        result.add_success(f"Valid audio file: {filepath.name}")
        return result

    def validate_directory(self, directory: Path) -> ValidationResult:
        """
        Validate all audio files in a directory.

        Parameters
        ----------
        directory : Path
            Path to directory containing audio files.

        Returns
        -------
        ValidationResult
            Combined validation result.
        """
        result = ValidationResult()

        if not directory.exists():
            result.add_error(f"Directory not found: {directory}")
            return result

        audio_files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.VALID_EXTENSIONS
        ]

        if not audio_files:
            result.add_warning(f"No audio files found in: {directory}")
            return result

        for filepath in audio_files:
            file_result = self.validate_file(filepath)
            result.merge(file_result)

        result.add_success(f"Validated {len(audio_files)} audio files")
        return result


class ValidationResult:
    """
    Container for validation results.

    This class stores validation messages and provides
    methods for querying validation status.

    Attributes
    ----------
    errors : list
        List of error messages.
    warnings : list
        List of warning messages.
    successes : list
        List of success messages.
    """

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.successes: list[str] = []

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        logger.error(f"Validation error: {message}")

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")

    def add_success(self, message: str) -> None:
        """Add a success message."""
        self.successes.append(message)
        logger.debug(f"Validation success: {message}")

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.successes.extend(other.successes)

    def __str__(self) -> str:
        """String representation of validation result."""
        lines = []
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  - {w}" for w in self.warnings)
        if self.is_valid:
            lines.append("Validation: PASSED")
        else:
            lines.append("Validation: FAILED")
        return "\n".join(lines)