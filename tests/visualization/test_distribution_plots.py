"""Unit tests for distribution plots module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestDistributionVisualizer:
    """Test suite for DistributionVisualizer class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.visualization.style.font_family = "serif"
        mock.visualization.style.font_serif = ["Times New Roman"]
        mock.visualization.style.figure_dpi = 100
        return mock

    def test_visualizer_initialization(self, mock_settings: MagicMock) -> None:
        """Test DistributionVisualizer initialization."""
        from voice_analysis.visualization.distribution_plots import DistributionVisualizer

        visualizer = DistributionVisualizer(mock_settings)

        assert visualizer.settings == mock_settings

    def test_plot_class_distribution(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        """Test plotting class distribution."""
        from voice_analysis.visualization.distribution_plots import DistributionVisualizer

        visualizer = DistributionVisualizer(mock_settings)

        before = pd.Series(["Control"] * 20 + ["Pathological"] * 10)
        after = pd.Series(["Control"] * 20 + ["Pathological"] * 20)

        save_path = tmp_path / "class_dist.svg"

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer.plot_class_distribution(before, after, save_path)

    def test_plot_feature_distributions(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        """Test plotting feature distributions."""
        from voice_analysis.visualization.distribution_plots import DistributionVisualizer

        visualizer = DistributionVisualizer(mock_settings)

        np.random.seed(42)
        before = pd.DataFrame(
            {
                "feature1": np.random.randn(20),
                "feature2": np.random.randn(20),
            }
        )
        after = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )

        save_path = tmp_path / "feature_dist.svg"

        with patch("matplotlib.pyplot.savefig"):
            with patch("matplotlib.pyplot.close"):
                visualizer.plot_feature_distributions(before, after, save_path)

    def test_clean_data_handles_infinity(self, mock_settings: MagicMock) -> None:
        """Test that _clean_data handles infinity values."""
        from voice_analysis.visualization.distribution_plots import DistributionVisualizer

        data = pd.DataFrame(
            {
                "col1": [1.0, np.inf, 3.0],
                "col2": [-np.inf, 2.0, 3.0],
            }
        )

        result = DistributionVisualizer._clean_data(data)

        assert np.isnan(result["col1"][1])
        assert np.isnan(result["col2"][0])

    def test_clean_data_series(self, mock_settings: MagicMock) -> None:
        """Test _clean_data with Series input."""
        from voice_analysis.visualization.distribution_plots import DistributionVisualizer

        data = pd.Series([1.0, np.inf, -np.inf, 4.0])

        result = DistributionVisualizer._clean_data(data)

        assert np.isnan(result[1])
        assert np.isnan(result[2])
