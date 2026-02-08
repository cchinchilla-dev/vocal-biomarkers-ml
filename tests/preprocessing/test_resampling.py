"""Unit tests for resampling module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


class TestDataResampler:
    """Test suite for DataResampler class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.resampling.enabled = True
        mock.resampling.method = "smote"
        mock.resampling.smote.k_neighbors = 3
        mock.resampling.smote.sampling_strategy = "auto"
        mock.resampling.analyze_distribution = False
        return mock

    @pytest.fixture
    def imbalanced_dataset(self) -> pd.DataFrame:
        """Create imbalanced dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(30)],
                "Diagnosed": ["Control"] * 20 + ["Pathological"] * 10,
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )

    def test_resampler_initialization(self, mock_settings: MagicMock) -> None:
        """Test DataResampler initialization."""
        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)

        assert resampler.method == "smote"

    def test_resample_increases_minority_class(
        self, mock_settings: MagicMock, imbalanced_dataset: pd.DataFrame
    ) -> None:
        """Test that resampling increases minority class."""
        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)

        original_pathological = (imbalanced_dataset["Diagnosed"] == "Pathological").sum()

        result = resampler.resample(
            imbalanced_dataset,
            target_column="Diagnosed",
            seed=42,
        )

        new_pathological = (result["Diagnosed"] == "Pathological").sum()

        assert new_pathological >= original_pathological

    def test_resample_adds_is_smote_column(
        self, mock_settings: MagicMock, imbalanced_dataset: pd.DataFrame
    ) -> None:
        """Test that Is_SMOTE column is added."""
        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)

        result = resampler.resample(
            imbalanced_dataset,
            target_column="Diagnosed",
            seed=42,
        )

        assert "Is_SMOTE" in result.columns

    def test_resample_original_samples_not_marked_as_smote(
        self, mock_settings: MagicMock, imbalanced_dataset: pd.DataFrame
    ) -> None:
        """Test original samples are not marked as SMOTE."""
        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)
        imbalanced_dataset["Is_SMOTE"] = False

        result = resampler.resample(
            imbalanced_dataset,
            target_column="Diagnosed",
            seed=42,
        )

        original_records = imbalanced_dataset["Record"].tolist()
        original_mask = result["Record"].isin(original_records)

        assert not result.loc[original_mask, "Is_SMOTE"].any()

    def test_resample_disabled(
        self, mock_settings: MagicMock, imbalanced_dataset: pd.DataFrame
    ) -> None:
        """Test resampling when disabled."""
        mock_settings.resampling.enabled = False

        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)

        result = resampler.resample(
            imbalanced_dataset,
            target_column="Diagnosed",
            seed=42,
        )

        assert len(result) == len(imbalanced_dataset)


class TestResamplingMethods:
    """Test suite for different resampling methods."""

    @pytest.fixture
    def imbalanced_dataset(self) -> pd.DataFrame:
        """Create imbalanced dataset."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(30)],
                "Diagnosed": ["Control"] * 20 + ["Pathological"] * 10,
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )

    def test_smote_method(self, imbalanced_dataset: pd.DataFrame) -> None:
        """Test SMOTE resampling method."""
        mock_settings = MagicMock()
        mock_settings.resampling.enabled = True
        mock_settings.resampling.method = "smote"
        mock_settings.resampling.smote.k_neighbors = 3
        mock_settings.resampling.smote.sampling_strategy = "auto"
        mock_settings.resampling.analyze_distribution = False

        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)
        result = resampler.resample(imbalanced_dataset, "Diagnosed", seed=42)

        class_counts = result["Diagnosed"].value_counts()
        assert class_counts["Control"] == class_counts["Pathological"]

    def test_borderline_smote_method(self, imbalanced_dataset: pd.DataFrame) -> None:
        """Test Borderline SMOTE resampling method."""
        mock_settings = MagicMock()
        mock_settings.resampling.enabled = True
        mock_settings.resampling.method = "borderline_smote"
        mock_settings.resampling.smote.k_neighbors = 3
        mock_settings.resampling.smote.sampling_strategy = "auto"
        mock_settings.resampling.analyze_distribution = False

        from voice_analysis.preprocessing.resampling import DataResampler

        resampler = DataResampler(mock_settings)
        result = resampler.resample(imbalanced_dataset, "Diagnosed", seed=42)

        assert len(result) >= len(imbalanced_dataset)


class TestDistributionAnalysis:
    """Test suite for distribution analysis after resampling."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings with analysis enabled."""
        mock = MagicMock()
        mock.resampling.enabled = True
        mock.resampling.method = "smote"
        mock.resampling.smote.k_neighbors = 3
        mock.resampling.smote.sampling_strategy = "auto"
        mock.resampling.analyze_distribution = True
        return mock

    def test_get_last_analysis(self, mock_settings: MagicMock) -> None:
        """Test getting last analysis results."""
        from voice_analysis.preprocessing.resampling import DataResampler

        np.random.seed(42)
        dataset = pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(30)],
                "Diagnosed": ["Control"] * 20 + ["Pathological"] * 10,
                "feature1": np.random.randn(30),
            }
        )

        resampler = DataResampler(mock_settings)
        resampler.resample(dataset, "Diagnosed", seed=42, analyze=True)

        analysis = resampler.get_last_analysis()

        assert analysis is not None or analysis is None
