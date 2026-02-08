"""Unit tests for cleaning module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from voice_analysis.preprocessing.cleaning import (
    DataCleaner,
    remove_correlated_features,
    remove_low_variance_features,
)


class TestRemoveLowVarianceFeatures:
    """Test suite for remove_low_variance_features function."""

    def test_removes_zero_variance_features(self) -> None:
        """Test that features with zero variance are removed."""
        df = pd.DataFrame(
            {
                "constant": [5, 5, 5, 5, 5],
                "varying": [1, 2, 3, 4, 5],
            }
        )

        result, removed = remove_low_variance_features(df, threshold=0.1)

        assert "constant" not in result.columns
        assert "varying" in result.columns
        assert "constant" in removed

    def test_respects_threshold(self) -> None:
        """Test that threshold is correctly applied."""
        df = pd.DataFrame(
            {
                "low_var": np.array([1.0, 1.01, 1.02, 0.99, 1.0]),  # var ≈ 0.0001
                "high_var": np.array([1, 10, 20, 5, 15]),  # var > 0.1
            }
        )

        result, removed = remove_low_variance_features(df, threshold=0.01)

        assert "low_var" not in result.columns
        assert "high_var" in result.columns

    def test_keeps_all_features_with_low_threshold(self) -> None:
        """Test that all features are kept with threshold of 0."""
        df = pd.DataFrame(
            {
                "a": [1, 1, 1, 1],
                "b": [1, 2, 3, 4],
            }
        )

        result, removed = remove_low_variance_features(df, threshold=0)

        assert len(result.columns) == 2
        assert len(removed) == 0

    def test_returns_tuple(self) -> None:
        """Test that function returns tuple of DataFrame and list."""
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = remove_low_variance_features(df, threshold=0.1)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], list)

    def test_handles_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        result, removed = remove_low_variance_features(df, threshold=0.1)

        assert result.empty
        assert removed == []


class TestRemoveCorrelatedFeatures:
    """Test suite for remove_correlated_features function."""

    def test_removes_highly_correlated_features(self) -> None:
        """Test that highly correlated features are removed."""
        np.random.seed(42)
        base = np.random.randn(100)

        df = pd.DataFrame(
            {
                "base": base,
                "correlated": base + np.random.randn(100) * 0.01,  # r ≈ 0.999
                "independent": np.random.randn(100),
            }
        )

        result, removed = remove_correlated_features(df, threshold=0.9)

        assert len(removed) == 1
        assert "independent" in result.columns

    def test_keeps_features_below_threshold(self) -> None:
        """Test that features below threshold are kept."""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
                "c": np.random.randn(50),
            }
        )

        result, removed = remove_correlated_features(df, threshold=0.9)

        assert len(result.columns) == 3
        assert len(removed) == 0

    def test_respects_threshold(self) -> None:
        """Test that correlation threshold is respected."""
        np.random.seed(42)
        base = np.random.randn(100)

        df = pd.DataFrame(
            {
                "a": base,
                "b": base * 0.8 + np.random.randn(100) * 0.6,
            }
        )

        result_high, _ = remove_correlated_features(df, threshold=0.99)
        assert len(result_high.columns) == 2

    def test_handles_single_column(self) -> None:
        """Test handling of single column DataFrame."""
        df = pd.DataFrame({"single": [1, 2, 3, 4, 5]})

        result, removed = remove_correlated_features(df, threshold=0.9)

        assert len(result.columns) == 1
        assert removed == []

    def test_only_processes_numeric_columns(self) -> None:
        """Test that only numeric columns are considered."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5],
                "string": ["a", "b", "c", "d", "e"],
            }
        )

        result, removed = remove_correlated_features(df, threshold=0.9)

        assert "string" in result.columns


class TestDataCleaner:
    """Test suite for DataCleaner class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for DataCleaner."""
        mock = MagicMock()
        mock.data_processing.cleaning.variance_threshold = 0.1
        mock.data_processing.cleaning.correlation_threshold = 0.9
        return mock

    @pytest.fixture
    def cleaner(self, mock_settings: MagicMock) -> DataCleaner:
        """Create a DataCleaner instance."""
        return DataCleaner(mock_settings)

    def test_initialization(self, cleaner: DataCleaner, mock_settings: MagicMock) -> None:
        """Test DataCleaner initialization."""
        assert cleaner.variance_threshold == 0.1
        assert cleaner.correlation_threshold == 0.9
        assert cleaner.settings == mock_settings

    def test_clean_removes_low_variance(self, cleaner: DataCleaner) -> None:
        """Test that clean removes low variance features."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c", "d", "e"],
                "Diagnosed": ["Control"] * 3 + ["Pathological"] * 2,
                "constant": [1, 1, 1, 1, 1],
                "varying": [1, 2, 3, 4, 5],
            }
        )

        result = cleaner.clean(df)

        assert "constant" not in result.columns
        assert "varying" in result.columns
        assert "Record" in result.columns  # Metadata preserved
        assert "Diagnosed" in result.columns

    def test_clean_removes_correlated(self, cleaner: DataCleaner) -> None:
        """Test that clean removes highly correlated features."""
        np.random.seed(42)
        base = np.random.randn(50)

        df = pd.DataFrame(
            {
                "Record": [f"r_{i}" for i in range(50)],
                "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
                "feature_a": base,
                "feature_b": base + np.random.randn(50) * 0.001,
                "feature_c": np.random.randn(50),
            }
        )

        result = cleaner.clean(df)

        correlated_remaining = sum(1 for col in ["feature_a", "feature_b"] if col in result.columns)
        assert correlated_remaining == 1
        assert "feature_c" in result.columns

    def test_clean_preserves_metadata_columns(self, cleaner: DataCleaner) -> None:
        """Test that metadata columns are preserved."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c"],
                "Diagnosed": ["Control", "Control", "Pathological"],
                "Gender": ["M", "F", "M"],
                "Age": [30, 40, 50],
                "feature": [1, 2, 3],
            }
        )

        result = cleaner.clean(df)

        assert "Record" in result.columns
        assert "Diagnosed" in result.columns
        assert "Gender" in result.columns
        assert "Age" in result.columns

    def test_clean_tracks_removed_features(self, cleaner: DataCleaner) -> None:
        """Test that removed features are tracked."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c", "d", "e"],
                "Diagnosed": ["Control"] * 3 + ["Pathological"] * 2,
                "constant": [1, 1, 1, 1, 1],
                "varying": [1, 2, 3, 4, 5],
            }
        )

        cleaner.clean(df)

        assert "constant" in cleaner.removed_features["low_variance"]

    def test_get_removal_report(self, cleaner: DataCleaner) -> None:
        """Test removal report generation."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c", "d", "e"],
                "Diagnosed": ["Control"] * 3 + ["Pathological"] * 2,
                "constant": [1, 1, 1, 1, 1],
                "varying": [1, 2, 3, 4, 5],
            }
        )

        cleaner.clean(df)
        report = cleaner.get_removal_report()

        assert isinstance(report, str)
        assert "Feature Removal Report" in report
        assert "Low Variance" in report

    def test_clean_handles_empty_features(self, cleaner: DataCleaner) -> None:
        """Test cleaning dataset with only metadata columns."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c"],
                "Diagnosed": ["Control", "Control", "Pathological"],
            }
        )

        result = cleaner.clean(df)

        assert len(result.columns) == 2

    def test_clean_returns_dataframe(self, cleaner: DataCleaner) -> None:
        """Test that clean returns a DataFrame."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b"],
                "Diagnosed": ["Control", "Pathological"],
                "feature": [1, 2],
            }
        )

        result = cleaner.clean(df)

        assert isinstance(result, pd.DataFrame)

    def test_clean_handles_is_smote_column(self, cleaner: DataCleaner) -> None:
        """Test that Is_SMOTE column is preserved as metadata."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c"],
                "Diagnosed": ["Control", "Control", "Pathological"],
                "Is_SMOTE": [False, False, True],
                "feature": [1, 2, 3],
            }
        )

        result = cleaner.clean(df)

        assert "Is_SMOTE" in result.columns

    def test_clean_with_all_constant_features(self, cleaner: DataCleaner) -> None:
        """Test cleaning when all feature columns are constant."""
        df = pd.DataFrame(
            {
                "Record": ["a", "b", "c"],
                "Diagnosed": ["Control", "Control", "Pathological"],
                "const1": [1, 1, 1],
                "const2": [2, 2, 2],
            }
        )

        result = cleaner.clean(df)

        assert "Record" in result.columns
        assert "Diagnosed" in result.columns
        assert "const1" not in result.columns
        assert "const2" not in result.columns
