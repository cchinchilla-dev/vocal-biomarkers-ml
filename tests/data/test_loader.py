"""Unit tests for data loader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings for data loader."""
        mock = MagicMock()
        mock.paths.data_dir = "data"
        mock.paths.results_dir = "data/results"
        mock.paths.recordings.control = "raw/control"
        mock.paths.recordings.pathological = "raw/pathological"
        mock.paths.biomechanical.control = "raw/bio/control.csv"
        mock.paths.biomechanical.pathological = "raw/bio/pathological.csv"
        mock.data_processing.records.control = ["all"]
        mock.data_processing.records.pathological = ["all"]
        mock.data_processing.audio.f0_min = 75
        mock.data_processing.audio.f0_max = 500
        mock.data_processing.audio.unit = "Hertz"
        mock.data_processing.audio.n_mfcc = 13
        return mock

    def test_loader_initialization(self, mock_settings: MagicMock) -> None:
        """Test DataLoader initialization."""
        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)

            assert loader.settings == mock_settings

    def test_load_processed_dataset(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Test loading processed dataset from CSV."""
        test_df = pd.DataFrame({
            "Record": ["a", "b"],
            "Diagnosed": ["Control", "Pathological"],
            "feature1": [1.0, 2.0],
        })

        datasets_path = tmp_path / "results" / "datasets"
        datasets_path.mkdir(parents=True)
        test_df.to_csv(datasets_path / "test.csv", index=False)

        mock_settings.paths.results_dir = str(tmp_path / "results")

        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)
            result = loader.load_processed_dataset("test.csv")

            assert len(result) == 2
            assert "Record" in result.columns

    def test_load_processed_dataset_not_found(
        self, mock_settings: MagicMock
    ) -> None:
        """Test loading non-existent dataset raises error."""
        mock_settings.paths.results_dir = "/nonexistent/path"
        mock_settings.paths.data_dir = "/nonexistent/data"

        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)

            with pytest.raises(FileNotFoundError):
                loader.load_processed_dataset("nonexistent.csv")

    def test_get_recordings_list_all(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        """Test getting all recordings from directory."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()
        (recordings_dir / "test1.wav").touch()
        (recordings_dir / "test2.wav").touch()
        (recordings_dir / "test3.mp3").touch()
        (recordings_dir / ".hidden.wav").touch()  # Should be ignored

        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)
            recordings = loader._get_recordings_list(recordings_dir, ["all"])

            assert len(recordings) == 3
            assert ".hidden.wav" not in recordings

    def test_get_recordings_list_specific(self, mock_settings: MagicMock, tmp_path: Path) -> None:
        """Test getting specific recordings."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()

        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)
            recordings = loader._get_recordings_list(
                recordings_dir, ["test1", "test2.wav"]
            )

            assert "test1.wav" in recordings
            assert "test2.wav" in recordings

    def test_merge_features(self, mock_settings: MagicMock) -> None:
        """Test merging acoustic and biomechanical features."""
        acoustic_df = pd.DataFrame({
            "Record": ["a", "b"],
            "acoustic_feature": [1.0, 2.0],
        })
        bio_df = pd.DataFrame({
            "Record": ["a", "b"],
            "bio_feature": [3.0, 4.0],
        })

        with patch("voice_analysis.data.loader.AcousticFeatureExtractor"):
            from voice_analysis.data.loader import DataLoader

            loader = DataLoader(mock_settings)
            merged = loader._merge_features(acoustic_df, bio_df, "Control")

            assert "acoustic_feature" in merged.columns
            assert "bio_feature" in merged.columns
            assert "Diagnosed" in merged.columns
            assert all(merged["Diagnosed"] == "Control")


class TestBiomechanicalDataLoader:
    """Test suite for BiomechanicalDataLoader class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        return mock

    def test_load_biomechanical_data(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Test loading biomechanical data from CSV."""
        test_df = pd.DataFrame({
            "Record": ["a", "b"],
            "Pr01": [100.0, 120.0],
            "Pr02": [0.5, 0.6],
        })
        filepath = tmp_path / "bio.csv"
        test_df.to_csv(filepath, index=False)

        from voice_analysis.data.loader import BiomechanicalDataLoader

        loader = BiomechanicalDataLoader(mock_settings)
        result = loader.load(filepath)

        assert len(result) == 2
        assert "Pr01" in result.columns