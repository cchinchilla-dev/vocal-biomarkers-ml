"""
Acoustic feature extraction module.

This module provides classes for extracting acoustic features
from audio recordings using Praat (via parselmouth) and librosa.
"""

import logging
import statistics
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.stats import kurtosis, skew

from ..config.settings import Settings

logger = logging.getLogger(__name__)


class AcousticFeatureExtractor:
    """
    Extractor for acoustic features from audio recordings.

    This class extracts various acoustic features including
    fundamental frequency (F0), formants, MFCCs, jitter, shimmer, and HNR.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    f0_min : int
        Minimum F0 frequency for analysis.
    f0_max : int
        Maximum F0 frequency for analysis.
    unit : str
        Unit for frequency measurements.
    n_mfcc : int
        Number of MFCC coefficients to extract.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        audio_config = settings.data_processing.audio
        self.f0_min = audio_config.f0_min
        self.f0_max = audio_config.f0_max
        self.unit = audio_config.unit
        self.n_mfcc = audio_config.n_mfcc

    def extract_from_recordings(
        self,
        recordings_dir: Path,
        recordings: list[str],
    ) -> pd.DataFrame:
        """
        Extract acoustic features from multiple recordings.

        Parameters
        ----------
        recordings_dir : Path
            Directory containing audio files.
        recordings : list
            List of recording filenames to process.

        Returns
        -------
        DataFrame
            Extracted features for all recordings.
        """
        all_features = []

        for recording in recordings:
            filepath = recordings_dir / recording
            if not filepath.exists():
                logger.warning(f"Recording not found: {filepath}")
                continue

            try:
                features = self.extract_from_file(filepath)
                record_name = filepath.stem
                features["Record"] = record_name
                all_features.append(features)
            except Exception as e:
                logger.error(f"Error processing {recording}: {e}")
                continue

        if not all_features:
            raise ValueError("No features extracted from any recording")

        df = pd.concat(all_features, ignore_index=True)

        # Move Record column to first position
        cols = ["Record"] + [c for c in df.columns if c != "Record"]
        return df[cols]

    def extract_from_file(self, filepath: Path) -> pd.DataFrame:
        """
        Extract all acoustic features from a single audio file.

        Parameters
        ----------
        filepath : Path
            Path to the audio file.

        Returns
        -------
        DataFrame
            Single-row DataFrame with all extracted features.
        """
        logger.debug(f"Extracting features from: {filepath}")

        # Extract Praat features
        praat_features = self._extract_praat_features(filepath)

        # Extract MFCC features
        mfcc_features = self._extract_mfcc_features(filepath)

        # Combine features
        return pd.concat([praat_features, mfcc_features], axis=1)

    def _extract_praat_features(self, filepath: Path) -> pd.DataFrame:
        """
        Extract features using Praat via parselmouth.

        Parameters
        ----------
        filepath : Path
            Path to the audio file.

        Returns
        -------
        DataFrame
            DataFrame with Praat-extracted features.
        """
        sound = parselmouth.Sound(str(filepath))

        # Convert stereo to mono if necessary
        if sound.n_channels == 2:
            sound = sound.convert_to_mono()

        features = {}

        # F0 analysis
        features.update(self._analyze_f0(sound))

        # Formant analysis
        features.update(self._analyze_formants(sound))

        # HNR analysis
        features.update(self._analyze_hnr(sound))

        # Jitter and Shimmer analysis
        features.update(self._analyze_jitter_shimmer(sound))

        return pd.DataFrame([features])

    def _analyze_f0(self, sound: parselmouth.Sound) -> dict:
        """
        Analyze fundamental frequency (F0).

        Parameters
        ----------
        sound : parselmouth.Sound
            Sound object to analyze.

        Returns
        -------
        dict
            Dictionary with F0 features.
        """
        pitch = call(sound, "To Pitch", 0, self.f0_min, self.f0_max)
        f0_values = pitch.selected_array["frequency"]

        features = {
            "F0_mean": call(pitch, "Get mean", 0, 0, self.unit),
            "F0_stdev": call(pitch, "Get standard deviation", 0, 0, self.unit),
            "F0_skew": skew(f0_values),
            "F0_kurtosis": kurtosis(f0_values),
        }

        # F0 delta features
        delta1, delta2 = self._compute_deltas(f0_values)
        features.update(self._compute_delta_stats(delta1, delta2, "F0"))

        return features

    def _analyze_formants(self, sound: parselmouth.Sound) -> dict:
        """
        Analyze formants F1 and F2.

        Parameters
        ----------
        sound : parselmouth.Sound
            Sound object to analyze.

        Returns
        -------
        dict
            Dictionary with formant features.
        """
        point_process = call(sound, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)
        num_points = call(point_process, "Get number of points")
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5500, 0.025, 50)

        features = {}

        for formant_num in [1, 2]:
            formant_values = []
            for point in range(num_points):
                time = call(point_process, "Get time from index", point + 1)
                value = call(formants, "Get value at time", formant_num, time, self.unit, "Linear")
                if value > 0:
                    formant_values.append(value)

            if formant_values:
                prefix = f"F{formant_num}"
                features[f"{prefix}_mean"] = np.mean(formant_values)
                features[f"{prefix}_stdev"] = statistics.stdev(formant_values)
                features[f"{prefix}_skew"] = skew(formant_values)
                features[f"{prefix}_kurtosis"] = kurtosis(formant_values)

                # Delta features
                delta1, delta2 = self._compute_deltas(formant_values)
                features.update(self._compute_delta_stats(delta1, delta2, prefix))

        return features

    def _analyze_hnr(self, sound: parselmouth.Sound) -> dict:
        """
        Analyze Harmonic-to-Noise Ratio.

        Parameters
        ----------
        sound : parselmouth.Sound
            Sound object to analyze.

        Returns
        -------
        dict
            Dictionary with HNR feature.
        """
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return {"HNR": call(harmonicity, "Get mean", 0, 0)}

    def _analyze_jitter_shimmer(self, sound: parselmouth.Sound) -> dict:
        """
        Analyze jitter and shimmer.

        Parameters
        ----------
        sound : parselmouth.Sound
            Sound object to analyze.

        Returns
        -------
        dict
            Dictionary with jitter and shimmer features.
        """
        point_process = call(sound, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)

        features = {
            # Jitter measures
            "Local_jitter": call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            "Local_absolute_jitter": call(
                point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3
            ),
            "RAP_jitter": call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            "PPQ5_jitter": call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            "DDP_jitter": call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            # Shimmer measures
            "Local_shimmer": call(
                [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
            "Local_dB_shimmer": call(
                [sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
            "APQ3_shimmer": call(
                [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
            "APQ5_shimmer": call(
                [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
            "APQ11_shimmer": call(
                [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
            "DDA_shimmer": call(
                [sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            ),
        }

        return features

    def _extract_mfcc_features(self, filepath: Path) -> pd.DataFrame:
        """
        Extract MFCC features using librosa.

        Parameters
        ----------
        filepath : Path
            Path to the audio file.

        Returns
        -------
        DataFrame
            DataFrame with MFCC features.
        """
        y, sr = librosa.load(str(filepath), sr=None)

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        features = {}

        for i in range(self.n_mfcc):
            idx = f"{i:02d}"

            # MFCC coefficients
            features[f"MFCC_{idx}"] = np.mean(mfccs[i])
            features[f"MFCC_{idx}_stdev"] = np.std(mfccs[i])
            features[f"MFCC_{idx}_skew"] = skew(mfccs[i])
            features[f"MFCC_{idx}_kurtosis"] = kurtosis(mfccs[i])

            # Delta coefficients
            features[f"MFCC_Delta_{idx}"] = np.mean(delta[i])
            features[f"MFCC_Delta_{idx}_stdev"] = np.std(delta[i])
            features[f"MFCC_Delta_{idx}_skew"] = skew(delta[i])
            features[f"MFCC_Delta_{idx}_kurtosis"] = kurtosis(delta[i])

            # Delta-delta coefficients
            features[f"MFCC_Delta2_{idx}"] = np.mean(delta2[i])
            features[f"MFCC_Delta2_{idx}_stdev"] = np.std(delta2[i])
            features[f"MFCC_Delta2_{idx}_skew"] = skew(delta2[i])
            features[f"MFCC_Delta2_{idx}_kurtosis"] = kurtosis(delta2[i])

        return pd.DataFrame([features])

    @staticmethod
    def _compute_deltas(values: list | np.ndarray) -> tuple[list, list]:
        """Compute first and second order deltas."""
        values = list(values)
        delta1 = [values[i] - values[i - 1] for i in range(1, len(values))]
        delta2 = [values[i] - values[i - 2] for i in range(2, len(values))]
        return delta1, delta2

    @staticmethod
    def _compute_delta_stats(delta1: list, delta2: list, prefix: str) -> dict:
        """Compute statistics for delta values."""
        features = {}

        if delta1:
            features[f"{prefix}_Delta1_mean"] = statistics.mean(delta1)
            features[f"{prefix}_Delta1_stdev"] = statistics.stdev(delta1) if len(delta1) > 1 else 0
            features[f"{prefix}_Delta1_skew"] = skew(delta1)
            features[f"{prefix}_Delta1_kurtosis"] = kurtosis(delta1)

        if delta2:
            features[f"{prefix}_Delta2_mean"] = statistics.mean(delta2)
            features[f"{prefix}_Delta2_stdev"] = statistics.stdev(delta2) if len(delta2) > 1 else 0
            features[f"{prefix}_Delta2_skew"] = skew(delta2)
            features[f"{prefix}_Delta2_kurtosis"] = kurtosis(delta2)

        return features
