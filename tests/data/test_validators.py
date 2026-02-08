"""Unit tests for validators module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestDataValidator:
    """Test suite for DataValidator class."""

    @pytest.fixture
    def validator(self):
        """Create DataValidator instance."""
        from voice_analysis.data.validators import DataValidator
        return DataValidator()

    @pytest.fixture
    def valid_dataset(self) -> pd.DataFrame:
        """Create valid dataset."""
        return pd.DataFrame({
            "Record": ["a", "b", "c"],
            "Diagnosed": ["Control", "Control", "Pathological"],
            "feature1": [1.0, 2.0, 3.0],
        })

    def test_validate_dataset_valid(self, validator, valid_dataset: pd.DataFrame) -> None:
        """Test validation of valid dataset."""
        result = validator.validate_dataset(valid_dataset)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_dataset_missing_required_columns(self, validator) -> None:
        """Test validation catches missing required columns."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0],
        })

        result = validator.validate_dataset(df)

        assert not result.is_valid
        assert any("Missing required columns" in e for e in result.errors)

    def test_validate_dataset_missing_values(self, validator) -> None:
        """Test validation detects missing values."""
        df = pd.DataFrame({
            "Record": ["a", "b", None],
            "Diagnosed": ["Control", "Pathological", "Control"],
            "feature1": [1.0, np.nan, 3.0],
        })

        result = validator.validate_dataset(df)

        assert len(result.warnings) > 0

    def test_validate_dataset_invalid_diagnosis(self, validator) -> None:
        """Test validation catches invalid diagnosis values."""
        df = pd.DataFrame({
            "Record": ["a", "b"],
            "Diagnosed": ["Control", "Invalid"],
            "feature1": [1.0, 2.0],
        })

        result = validator.validate_dataset(df)

        assert any("Invalid diagnosis values" in e for e in result.errors)

    def test_validate_dataset_duplicates(self, validator) -> None:
        """Test validation detects duplicate records."""
        df = pd.DataFrame({
            "Record": ["a", "a", "b"],
            "Diagnosed": ["Control", "Control", "Pathological"],
            "feature1": [1.0, 1.0, 2.0],
        })

        result = validator.validate_dataset(df)

        assert any("duplicate" in w.lower() for w in result.warnings)

    def test_check_numeric_columns(self, validator) -> None:
        """Test numeric column validation."""
        df = pd.DataFrame({
            "Record": ["a", "b"],
            "Diagnosed": ["Control", "Pathological"],
            "feature1": [1.0, np.inf],
        })

        result = validator.validate_dataset(df)

        assert len(result.warnings) > 0


class TestValidationResult:
    """Test suite for ValidationResult class."""

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult creation."""
        from voice_analysis.data.validators import ValidationResult

        result = ValidationResult()

        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error(self) -> None:
        """Test adding error to ValidationResult."""
        from voice_analysis.data.validators import ValidationResult

        result = ValidationResult()
        result.add_error("Test error")

        assert not result.is_valid
        assert "Test error" in result.errors

    def test_add_warning(self) -> None:
        """Test adding warning to ValidationResult."""
        from voice_analysis.data.validators import ValidationResult

        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid
        assert "Test warning" in result.warnings

    def test_add_success(self) -> None:
        """Test adding success message."""
        from voice_analysis.data.validators import ValidationResult

        result = ValidationResult()
        result.add_success("Test success")

        assert result.is_valid
        assert "Test success" in result.successes


class TestFileValidation:
    @pytest.fixture
    def validator(self):
        from voice_analysis.data.validators import AudioFileValidator
        return AudioFileValidator()

    def test_validate_audio_file_extension(self, validator, tmp_path: Path) -> None:
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"\x00")
        result = validator.validate_file(wav_file)
        assert result.is_valid

        txt_file = tmp_path / "test.txt"
        txt_file.write_bytes(b"\x00")
        result = validator.validate_file(txt_file)
        assert not result.is_valid