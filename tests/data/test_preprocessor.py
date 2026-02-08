"""Unit tests for preprocessor module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        return MagicMock()

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset."""
        np.random.seed(42)
        return pd.DataFrame({
            "Record": ["a", "b", "c", "d", "e", "f"],
            "Diagnosed": ["Control", "Control", "Control",
                         "Pathological", "Pathological", "Pathological"],
            "feature1": np.random.randn(6) * 10 + 100,
            "feature2": np.random.randn(6) * 0.1,
        })

    def test_preprocessor_initialization(self, mock_settings: MagicMock) -> None:
        """Test DataPreprocessor initialization."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)

        assert preprocessor.settings == mock_settings
        assert preprocessor._is_fitted is False

    def test_prepare_for_training(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test prepare_for_training method."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)
        X, y, record_ids = preprocessor.prepare_for_training(sample_dataset)

        assert len(X) == len(sample_dataset)
        assert len(y) == len(sample_dataset)
        assert "Record" not in X.columns
        assert "Diagnosed" not in X.columns
        assert preprocessor._is_fitted is True

    def test_prepare_for_training_encodes_labels(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that labels are encoded."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)
        X, y, _ = preprocessor.prepare_for_training(sample_dataset)

        assert set(y.unique()) == {0, 1}

    def test_prepare_for_training_standardizes_features(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test that features are standardized."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)
        X, _, _ = preprocessor.prepare_for_training(sample_dataset)

        for col in X.columns:
            assert abs(X[col].mean()) < 0.5
            assert 0.5 < X[col].std() < 1.5

    def test_inverse_transform_labels(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test inverse_transform_labels method."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)
        _, y_encoded, _ = preprocessor.prepare_for_training(sample_dataset)

        y_original = preprocessor.inverse_transform_labels(y_encoded.values)

        assert set(y_original) == {"Control", "Pathological"}

    def test_prepare_for_training_without_fit(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test prepare_for_training with fit=False."""
        from voice_analysis.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(mock_settings)

        preprocessor.prepare_for_training(sample_dataset, fit=True)

        X, y, _ = preprocessor.prepare_for_training(sample_dataset, fit=False)

        assert len(X) == len(sample_dataset)


class TestDatasetMergerStaticMethods:
    """Test suite for DatasetMerger static methods."""

    def test_merge_on_record(self) -> None:
        from voice_analysis.data.preprocessor import DatasetMerger
        df1 = pd.DataFrame({"Record": ["a", "b"], "col1": [1, 2]})
        df2 = pd.DataFrame({"Record": ["a", "b"], "col2": [3, 4]})
        result = DatasetMerger.merge_on_record(df1, df2)
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert len(result) == 2

    def test_concatenate_groups(self) -> None:
        from voice_analysis.data.preprocessor import DatasetMerger
        control = pd.DataFrame({"Record": ["a"], "Diagnosed": ["Control"]})
        pathological = pd.DataFrame({"Record": ["b"], "Diagnosed": ["Pathological"]})
        result = DatasetMerger.concatenate_groups(control, pathological)
        assert len(result) == 2
        assert set(result["Diagnosed"]) == {"Control", "Pathological"}

    def test_reorder_columns(self) -> None:
        from voice_analysis.data.preprocessor import DatasetMerger
        df = pd.DataFrame({
            "feature1": [1], "Record": ["a"],
            "feature2": [2], "Diagnosed": ["Control"],
        })
        result = DatasetMerger.reorder_columns(df)
        assert list(result.columns)[:2] == ["Record", "Diagnosed"]

    def test_reorder_columns_custom_priority(self) -> None:
        from voice_analysis.data.preprocessor import DatasetMerger
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
        result = DatasetMerger.reorder_columns(df, priority_columns=["col_c", "col_a"])
        assert list(result.columns)[:2] == ["col_c", "col_a"]

