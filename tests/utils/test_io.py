"""Unit tests for IO utilities module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestResultsManager:
    """Test suite for ResultsManager class."""

    @pytest.fixture
    def mock_output_config(self) -> MagicMock:
        """Create mock output configuration."""
        mock = MagicMock()
        mock.datasets = "datasets"
        mock.features = "features"
        mock.metrics = "metrics"
        mock.visualizations = "visualizations"
        return mock

    def test_results_manager_creates_directories(
        self, tmp_path: Path, mock_output_config: MagicMock
    ) -> None:
        """Test that ResultsManager creates necessary directories."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        assert (tmp_path / "datasets").exists()
        assert (tmp_path / "features").exists()
        assert (tmp_path / "metrics").exists()
        assert (tmp_path / "visualizations").exists()

    def test_save_dataset(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test saving dataset to CSV."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        filepath = manager.save_dataset(df, "test.csv")

        assert filepath.exists()
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2

    def test_load_dataset(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test loading dataset from CSV."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        # Save first
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        manager.save_dataset(df, "test.csv")

        # Load
        loaded = manager.load_dataset("test.csv")

        assert len(loaded) == 2
        assert "col1" in loaded.columns

    def test_load_dataset_not_found(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test loading non-existent dataset raises error."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        with pytest.raises(FileNotFoundError):
            manager.load_dataset("nonexistent.csv")

    def test_save_metrics(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test saving metrics to CSV."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        df = pd.DataFrame({"metric": ["accuracy"], "value": [0.85]})
        filepath = manager.save_metrics(df, "metrics.csv")

        assert filepath.exists()

    def test_save_features(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test saving features to CSV."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)

        df = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.5, 0.3]})
        filepath = manager.save_features(df, "features.csv")

        assert filepath.exists()

    def test_get_visualization_path(self, tmp_path: Path, mock_output_config: MagicMock) -> None:
        """Test getting visualization directory path."""
        from voice_analysis.utils.io import ResultsManager

        manager = ResultsManager(tmp_path, mock_output_config)
        vis_path = manager.get_visualization_path()

        assert vis_path == tmp_path / "visualizations"


class TestIOFunctions:
    """Test suite for standalone IO functions."""

    def test_save_dataframe(self, tmp_path: Path) -> None:
        """Test save_dataframe function."""
        from voice_analysis.utils.io import save_dataframe

        df = pd.DataFrame({"a": [1, 2], "b": [3.14159, 2.71828]})
        filepath = tmp_path / "test.csv"

        save_dataframe(df, filepath, float_format="%.2f")

        assert filepath.exists()

        loaded = pd.read_csv(filepath)
        assert loaded["b"][0] == pytest.approx(3.14, abs=0.01)

    def test_load_dataframe(self, tmp_path: Path) -> None:
        """Test load_dataframe function."""
        from voice_analysis.utils.io import load_dataframe, save_dataframe

        df = pd.DataFrame({"a": [1, 2]})
        filepath = tmp_path / "test.csv"
        save_dataframe(df, filepath)

        loaded = load_dataframe(filepath)

        assert len(loaded) == 2

    def test_save_dataframe_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save_dataframe creates parent directories."""
        from voice_analysis.utils.io import save_dataframe

        df = pd.DataFrame({"a": [1]})
        filepath = tmp_path / "nested" / "dir" / "test.csv"

        save_dataframe(df, filepath)

        assert filepath.exists()
