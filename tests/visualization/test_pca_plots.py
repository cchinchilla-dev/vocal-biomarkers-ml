"""Unit tests for PCA plots module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestPCAVisualizer:
    """Test suite for PCAVisualizer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.visualization.style.font_family = "serif"
        mock.visualization.style.font_serif = ["Times New Roman"]
        mock.visualization.style.figure_dpi = 100
        mock.visualization.pca.n_components = [2, 3]
        mock.visualization.pca.kernels = ["linear", "rbf"]
        return mock

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset for PCA."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(30)],
                "Diagnosed": ["Control"] * 15 + ["Pathological"] * 15,
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
                "feature3": np.random.randn(30),
            }
        )

    def test_visualizer_initialization(self, mock_settings: MagicMock) -> None:
        """Test PCAVisualizer initialization."""
        from voice_analysis.visualization.pca_plots import PCAVisualizer

        visualizer = PCAVisualizer(mock_settings)

        assert visualizer.settings == mock_settings

    def test_get_feature_columns(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame
    ) -> None:
        """Test getting feature columns."""
        from voice_analysis.visualization.pca_plots import PCAVisualizer

        visualizer = PCAVisualizer(mock_settings)
        features = visualizer._get_feature_columns(sample_dataset)

        assert "Record" not in features
        assert "Diagnosed" not in features
        assert "feature1" in features
        assert "feature2" in features

    def test_plot_creates_files(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test that plot creates output files."""
        from voice_analysis.visualization.pca_plots import PCAVisualizer

        visualizer = PCAVisualizer(mock_settings)

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer.plot(sample_dataset, tmp_path)

    def test_plot_linear_pca_2d(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test 2D linear PCA plotting."""
        from voice_analysis.visualization.pca_plots import PCAVisualizer

        visualizer = PCAVisualizer(mock_settings)
        features = visualizer._get_feature_columns(sample_dataset)

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer._plot_linear_pca(sample_dataset, features, 2, tmp_path)

    def test_plot_linear_pca_3d(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test 3D linear PCA plotting."""
        from voice_analysis.visualization.pca_plots import PCAVisualizer

        visualizer = PCAVisualizer(mock_settings)
        features = visualizer._get_feature_columns(sample_dataset)

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer._plot_linear_pca(sample_dataset, features, 3, tmp_path)


class TestCheckPCAConditions:
    """Test suite for check_pca_conditions function."""

    def test_valid_dataset_passes(self) -> None:
        """Test that valid dataset passes PCA conditions."""
        from voice_analysis.visualization.pca_plots import check_pca_conditions

        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        assert check_pca_conditions(df) is True

    def test_missing_values_fails(self) -> None:
        """Test that missing values fail PCA conditions."""
        from voice_analysis.visualization.pca_plots import check_pca_conditions

        df = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": [2.0, 3.0, 4.0],
            }
        )

        assert check_pca_conditions(df) is False

    def test_insufficient_unique_rows_fails(self) -> None:
        """Test that insufficient unique rows fail PCA conditions."""
        from voice_analysis.visualization.pca_plots import check_pca_conditions

        df = pd.DataFrame(
            {
                "feature1": [1.0, 1.0, 1.0],
                "feature2": [2.0, 2.0, 2.0],
            }
        )

        assert check_pca_conditions(df) is False
