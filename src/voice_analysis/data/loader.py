"""
Data loading module for voice analysis.

This module handles loading audio recordings and biomechanical data
from various sources and formats.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from ..config.settings import Settings
from ..features.acoustic import AcousticFeatureExtractor
from .validators import DataValidator

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loader for voice analysis data.

    This class handles loading audio recordings and biomechanical
    marker data, combining them into a unified dataset.

    Parameters
    ----------
    settings : Settings
        Configuration settings.

    Attributes
    ----------
    settings : Settings
        Configuration settings.
    acoustic_extractor : AcousticFeatureExtractor
        Extractor for acoustic features from audio.
    validator : DataValidator
        Validator for data integrity checks.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.acoustic_extractor = AcousticFeatureExtractor(settings)
        self.validator = DataValidator()

    def analyze_recordings(self) -> pd.DataFrame:
        """
        Analyze all audio recordings and combine with biomechanical data.

        Returns
        -------
        DataFrame
            Combined dataset with acoustic and biomechanical features.
        """
        logger.info("Analyzing audio recordings...")

        # Load biomechanical data
        control_bio = self._load_biomechanical_data("control")
        pathological_bio = self._load_biomechanical_data("pathological")

        # Extract acoustic features from recordings
        control_acoustic = self._extract_acoustic_features("control")
        pathological_acoustic = self._extract_acoustic_features("pathological")

        # Merge acoustic and biomechanical data
        control_dataset = self._merge_features(
            control_acoustic, control_bio, label="Control"
        )
        pathological_dataset = self._merge_features(
            pathological_acoustic, pathological_bio, label="Pathological"
        )

        # Combine datasets
        dataset = pd.concat([control_dataset, pathological_dataset], ignore_index=True)

        # Reorder columns
        dataset = self._reorder_columns(dataset)

        # Remove missing values
        dataset = dataset.dropna()

        logger.info(f"Dataset created with shape: {dataset.shape}")
        return dataset

    def load_processed_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load a previously processed dataset from CSV.

        Parameters
        ----------
        filename : str
            Name of the CSV file to load.

        Returns
        -------
        DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        filepath = Path(self.settings.paths.results_dir) / "datasets" / filename

        if not filepath.exists():
            # Try in analysis subdirectory for backward compatibility
            filepath = Path(self.settings.paths.data_dir) / "analysis" / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filename}")

        logger.info(f"Loading dataset from: {filepath}")
        return pd.read_csv(filepath)

    def _load_biomechanical_data(self, group: str) -> pd.DataFrame:
        """
        Load biomechanical marker data for a specific group.

        Parameters
        ----------
        group : str
            Group identifier ('control' or 'pathological').

        Returns
        -------
        DataFrame
            Biomechanical marker data.
        """
        data_dir = Path(self.settings.paths.data_dir)

        if group == "control":
            filepath = data_dir / self.settings.paths.biomechanical.control
        else:
            filepath = data_dir / self.settings.paths.biomechanical.pathological

        if not filepath.exists():
            raise FileNotFoundError(f"Biomechanical data not found: {filepath}")

        logger.debug(f"Loading biomechanical data from: {filepath}")
        return pd.read_csv(filepath)

    def _extract_acoustic_features(self, group: str) -> pd.DataFrame:
        """
        Extract acoustic features from audio recordings.

        Parameters
        ----------
        group : str
            Group identifier ('control' or 'pathological').

        Returns
        -------
        DataFrame
            Extracted acoustic features.
        """
        data_dir = Path(self.settings.paths.data_dir)

        if group == "control":
            recordings_dir = data_dir / self.settings.paths.recordings.control
            records_config = self.settings.data_processing.records.control
        else:
            recordings_dir = data_dir / self.settings.paths.recordings.pathological
            records_config = self.settings.data_processing.records.pathological

        if not recordings_dir.exists():
            raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

        # Get list of recordings to process
        recordings = self._get_recordings_list(recordings_dir, records_config)

        logger.info(f"Extracting features from {len(recordings)} {group} recordings")

        return self.acoustic_extractor.extract_from_recordings(
            recordings_dir, recordings
        )

    def _get_recordings_list(
        self, recordings_dir: Path, records_config: list[str]
    ) -> list[str]:
        """Get list of recording files to process."""
        if records_config == ["all"] or "all" in records_config:
            # Get all audio files in directory
            recordings = [
                f.name
                for f in recordings_dir.iterdir()
                if f.is_file()
                and not f.name.startswith(".")
                and f.suffix.lower() in [".wav", ".mp3", ".flac"]
            ]
        else:
            # Use specified recordings
            recordings = [
                r if r.endswith((".wav", ".mp3", ".flac")) else f"{r}.wav"
                for r in records_config
            ]

        return recordings

    def _merge_features(
        self,
        acoustic_df: pd.DataFrame,
        biomechanical_df: pd.DataFrame,
        label: str,
    ) -> pd.DataFrame:
        """
        Merge acoustic and biomechanical features.

        Parameters
        ----------
        acoustic_df : DataFrame
            Acoustic features dataset.
        biomechanical_df : DataFrame
            Biomechanical features dataset.
        label : str
            Diagnosis label to assign.

        Returns
        -------
        DataFrame
            Merged dataset with diagnosis label.
        """
        # Merge on Record column
        merged = pd.merge(acoustic_df, biomechanical_df, on="Record", how="inner")

        # Add diagnosis label
        merged["Diagnosed"] = label

        return merged

    def _reorder_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder columns to have metadata first.

        Parameters
        ----------
        dataset : DataFrame
            Dataset to reorder.

        Returns
        -------
        DataFrame
            Dataset with reordered columns.
        """
        priority_cols = ["Record", "Diagnosed", "Gender", "Age"]
        existing_priority = [c for c in priority_cols if c in dataset.columns]
        other_cols = [c for c in dataset.columns if c not in priority_cols]

        return dataset[existing_priority + other_cols]


class BiomechanicalDataLoader:
    """
    Specialized loader for biomechanical marker data.

    This class handles loading and preprocessing of biomechanical
    markers extracted from the Voice Clinical Systems App.
    """

    # Biomechanical marker descriptions
    MARKER_DESCRIPTIONS = {
        "Pr01": "Fundamental frequency (Hz)",
        "Pr02": "Ratio of cycles in closing",
        "Pr03": "Asymmetry",
        "Pr04": "Duration of closed phase (%)",
        "Pr05": "Duration of open phase (%)",
        "Pr06": "Duration opening (%)",
        "Pr07": "Duration closing (%)",
        "Pr08": "Stress Ratio (UR)",
        "Pr09": "Glottic closure force",
        "Pr10": "Efficiency index",
        "Pr11": "Gap width",
        "Pr12": "Gap size (%)",
        "Pr13": "Instability index",
        "Pr14": "Index variation in amplitude",
        "Pr15": "Vibration locking index",
        "Pr16": "Amplitude index",
        "Pr17": "Index OM closed phase",
        "Pr18": "Index OM opening phase",
        "Pr19": "Adec. OM Closed",
        "Pr20": "Adec OM Opening",
        "Pr21": "Structural imbalance index",
        "Pr22": "Mass alteration index",
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    def load(self, filepath: Path) -> pd.DataFrame:
        """
        Load biomechanical data from CSV file.

        Parameters
        ----------
        filepath : Path
            Path to the CSV file.

        Returns
        -------
        DataFrame
            Biomechanical marker data.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Biomechanical data not found: {filepath}")

        df = pd.read_csv(filepath)

        # Validate expected columns
        self._validate_columns(df)

        return df

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that expected columns are present.

        Parameters
        ----------
        df : DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        required = ["Record"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def get_marker_description(self, marker: str) -> str:
        """
        Get description for a biomechanical marker.

        Parameters
        ----------
        marker : str
            Marker code (e.g., 'Pr01').

        Returns
        -------
        str
            Description of the marker.
        """
        return self.MARKER_DESCRIPTIONS.get(marker, "Unknown marker")