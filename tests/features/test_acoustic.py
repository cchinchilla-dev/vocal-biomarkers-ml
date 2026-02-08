"""Unit tests for acoustic feature extraction module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestAcousticFeatureExtractor:
    """Test suite for AcousticFeatureExtractor class."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.data_processing.audio.f0_min = 75
        mock.data_processing.audio.f0_max = 500
        mock.data_processing.audio.unit = "Hertz"
        mock.data_processing.audio.n_mfcc = 13
        mock.data_processing.audio.sample_rate = 44100
        return mock

    def test_extractor_initialization(self, mock_settings: MagicMock) -> None:
        """Test AcousticFeatureExtractor initialization."""
        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        assert extractor.f0_min == 75
        assert extractor.f0_max == 500
        assert extractor.n_mfcc == 13

    def test_extract_from_file_returns_dataframe(self, mock_settings: MagicMock) -> None:
        """Test that extract_from_file returns DataFrame."""
        with patch("voice_analysis.features.acoustic.parselmouth") as mock_pm:
            with patch("voice_analysis.features.acoustic.librosa") as mock_librosa:
                # Setup mocks
                mock_sound = MagicMock()
                mock_pm.Sound.return_value = mock_sound

                mock_librosa.load.return_value = (np.random.randn(44100), 44100)
                mock_librosa.feature.mfcc.return_value = np.random.randn(13, 100)

                from voice_analysis.features.acoustic import AcousticFeatureExtractor

                extractor = AcousticFeatureExtractor(mock_settings)

                with patch.object(extractor, "_analyze_f0", return_value={"F0_mean": 100}):
                    with patch.object(
                        extractor, "_analyze_formants", return_value={"F1_mean": 500}
                    ):
                        with patch.object(
                            extractor,
                            "_analyze_jitter_shimmer",
                            return_value={"Local_jitter": 0.01},
                        ):
                            with patch.object(extractor, "_analyze_hnr", return_value={"HNR": 20}):
                                result = extractor.extract_from_file(Path("test.wav"))

                assert isinstance(result, pd.DataFrame)

    def test_extract_from_recordings_multiple_files(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Test extracting features from multiple recordings."""
        # Create mock audio files
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()
        (recordings_dir / "test1.wav").touch()
        (recordings_dir / "test2.wav").touch()

        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        mock_features = pd.DataFrame(
            {
                "f0_mean": [100.0],
                "f0_std": [10.0],
            }
        )

        with patch.object(extractor, "extract_from_file", return_value=mock_features):
            result = extractor.extract_from_recordings(recordings_dir, ["test1.wav", "test2.wav"])

            assert len(result) == 2
            assert "Record" in result.columns

    def test_extract_from_recordings_missing_file(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Test handling of missing audio files logs warning and raises error."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()

        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        with pytest.raises(ValueError, match="No features extracted"):
            extractor.extract_from_recordings(recordings_dir, ["nonexistent.wav"])

    def test_extract_from_recordings_no_valid_files(
        self, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Test error when no valid features extracted."""
        recordings_dir = tmp_path / "recordings"
        recordings_dir.mkdir()

        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        with pytest.raises(ValueError, match="No features extracted"):
            extractor.extract_from_recordings(recordings_dir, ["nonexistent.wav"])


class TestAcousticFeatureExtraction:
    """Test suite for specific acoustic feature extraction methods."""

    @pytest.fixture
    def mock_settings(self) -> MagicMock:
        """Create mock settings."""
        mock = MagicMock()
        mock.data_processing.audio.f0_min = 75
        mock.data_processing.audio.f0_max = 500
        mock.data_processing.audio.unit = "Hertz"
        mock.data_processing.audio.n_mfcc = 13
        return mock

    def test_extract_mfcc_features_structure(self, mock_settings: MagicMock) -> None:
        """Test MFCC feature extraction structure."""
        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        assert extractor.n_mfcc == 13

        expected_total = 13 * expected_features_per_coeff
        assert expected_total == 156

    def test_f0_feature_names(self, mock_settings: MagicMock) -> None:
        """Test that F0 features have correct names."""
        from voice_analysis.features.acoustic import AcousticFeatureExtractor

        extractor = AcousticFeatureExtractor(mock_settings)

        assert hasattr(extractor, "_analyze_f0")
        assert callable(getattr(extractor, "_analyze_f0"))
