"""Unit tests for standardization module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestDataStandardizer:
    """Test suite for DataStandardizer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        return mock

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset with different scales."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Record": ["a", "b", "c", "d", "e"],
                "Diagnosed": ["Control", "Control", "Pathological", "Pathological", "Control"],
                "feature1": np.array([100, 200, 300, 400, 500]),  # Large scale
                "feature2": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # Small scale
            }
        )

    def test_standardizer_initialization(self, mock_settings: MagicMock) -> None:
        """Test DataStandardizer initialization."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)

        assert standardizer.is_fitted is False

    def test_fit_transform(self, mock_settings: MagicMock, sample_dataset: pd.DataFrame) -> None:
        """Test fit_transform method."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)
        result = standardizer.fit_transform(sample_dataset)

        feature_cols = ["feature1", "feature2"]
        for col in feature_cols:
            assert abs(result[col].mean()) < 0.1
            assert abs(result[col].std(ddof=0) - 1.0) < 0.1

        assert list(result["Record"]) == list(sample_dataset["Record"])
        assert list(result["Diagnosed"]) == list(sample_dataset["Diagnosed"])

    def test_transform_after_fit(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test transform on new data after fitting."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)

        standardizer.fit(sample_dataset)

        new_data = pd.DataFrame(
            {
                "Record": ["x", "y"],
                "Diagnosed": ["Control", "Pathological"],
                "feature1": [150, 350],
                "feature2": [0.15, 0.35],
            }
        )

        result = standardizer.transform(new_data)

        assert standardizer.is_fitted
        assert len(result) == 2

    def test_transform_without_fit_raises(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test transform without fit raises error."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)

        with pytest.raises(RuntimeError, match="must be fitted"):
            standardizer.transform(sample_dataset)

    def test_inverse_transform(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test inverse_transform recovers original values."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)

        standardized = standardizer.fit_transform(sample_dataset)
        recovered = standardizer.inverse_transform(standardized)

        np.testing.assert_array_almost_equal(
            recovered["feature1"].values,
            sample_dataset["feature1"].values,
            decimal=5,
        )

    def test_standardize_convenience_method(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test standardize convenience method."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        standardizer = DataStandardizer(mock_settings)
        result = standardizer.standardize(sample_dataset)

        assert standardizer.is_fitted
        assert len(result) == len(sample_dataset)

    def test_standardize_split(self, mock_settings: MagicMock) -> None:
        """Test standardize_split for train/test split."""
        from voice_analysis.preprocessing.standardization import DataStandardizer

        np.random.seed(42)

        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(30) * 10 + 100,
                "feature2": np.random.randn(30) * 0.1,
            }
        )

        X_test = pd.DataFrame(
            {
                "feature1": np.random.randn(10) * 10 + 100,
                "feature2": np.random.randn(10) * 0.1,
            }
        )

        standardizer = DataStandardizer(MagicMock())
        X_train_std, X_test_std = standardizer.standardize_split(X_train, X_test)

        for col in X_train_std.columns:
            assert abs(X_train_std[col].mean()) < 0.1

        assert len(X_test_std) == len(X_test)
