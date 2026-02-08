"""Unit tests for ROC curves module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestROCVisualizer:
    """Test suite for ROCVisualizer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.visualization.style.font_family = "serif"
        mock.visualization.style.font_serif = ["Times New Roman"]
        mock.visualization.style.figure_dpi = 100
        mock.visualization.roc.enabled = True
        mock.visualization.roc.plot_folds = True
        mock.visualization.roc.plot_mean = True
        mock.visualization.roc.plot_std = True
        mock.models.enabled_models = ["RandomForest"]
        mock.models.cross_validation.n_folds = 3
        mock.models.cross_validation.shuffle = True
        mock.reproducibility.seeds = [42]
        return mock

    @pytest.fixture
    def sample_dataset(self) -> pd.DataFrame:
        """Create sample dataset for ROC curves."""
        np.random.seed(42)
        n = 50

        control = np.random.randn(25, 3) - 1
        pathological = np.random.randn(25, 3) + 1

        return pd.DataFrame(
            {
                "Record": [f"s_{i}" for i in range(n)],
                "Diagnosed": ["Control"] * 25 + ["Pathological"] * 25,
                "feature1": np.vstack([control, pathological])[:, 0],
                "feature2": np.vstack([control, pathological])[:, 1],
                "feature3": np.vstack([control, pathological])[:, 2],
            }
        )

    def test_visualizer_initialization(self, mock_settings: MagicMock) -> None:
        """Test ROCVisualizer initialization."""
        from voice_analysis.visualization.roc_curves import ROCVisualizer

        visualizer = ROCVisualizer(mock_settings)

        assert visualizer.settings == mock_settings

    def test_prepare_data(self, mock_settings: MagicMock, sample_dataset: pd.DataFrame) -> None:
        """Test data preparation for ROC curves."""
        from voice_analysis.visualization.roc_curves import ROCVisualizer

        visualizer = ROCVisualizer(mock_settings)
        X, y = visualizer._prepare_data(sample_dataset)

        assert X.shape[0] == len(sample_dataset)
        assert len(y) == len(sample_dataset)
        assert set(np.unique(y)) == {0, 1}

    def test_plot_creates_output(
        self, mock_settings: MagicMock, sample_dataset: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test that plot creates output file."""
        from voice_analysis.visualization.roc_curves import ROCVisualizer

        visualizer = ROCVisualizer(mock_settings)
        save_path = tmp_path / "roc.svg"

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer.plot(sample_dataset, save_path, seed=42)


class TestPlotROCComparison:
    """Test suite for plot_roc_comparison function."""

    def test_plot_roc_comparison(self, tmp_path: Path) -> None:
        """Test ROC comparison plotting."""
        from voice_analysis.visualization.roc_curves import plot_roc_comparison

        results = {
            "RandomForest": (np.array([0, 0.2, 0.5, 1]), np.array([0, 0.6, 0.8, 1]), 0.85),
            "SVM": (np.array([0, 0.3, 0.6, 1]), np.array([0, 0.5, 0.7, 1]), 0.80),
        }

        save_path = tmp_path / "roc_comparison.svg"

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                plot_roc_comparison(results, save_path)
